"""Multi-GPU (DDP) fine-tuning of the SAM encoder + AIS decoder on ALL labeled Candida data.

This is the *final* training run: unlike ``candida_multigpu_ais.py`` (which does k-fold
cross-validation to estimate generalization), this script trains a single model on **every**
labeled image. Use it once you are happy with the CV results and want the production model.

Design
------
* **All images train.** Every discovered raw/label pair goes into the training set, with the
  same augmentation pipeline the CV script uses (``build_train_transform``).
* **Validation == fixed crops of the training images.** ``DefaultTrainer`` still needs a
  ``val_loader`` to select ``best.pt`` and drive ``ReduceLROnPlateau``. There is no held-out
  data here, so validation uses one fixed, cell-rich patch per image
  (``fixed_crop_val_dataset``) with augmentation disabled. The val metric is therefore a
  checkpoint selector / plateau signal on the training data, **not** a generalization estimate
  (the CV script already provides that).
* **Reuses the CV building blocks.** ``build_unetr_model``, ``RankTensorboardLogger``,
  ``fixed_crop_val_dataset`` and ``build_train_transform`` are imported from
  ``candida_multigpu_ais`` -- no duplicated model / augmentation / logger code.
* **Auto-export.** After training, ``best.pt`` (a DDP checkpoint with ``module.``-prefixed keys)
  is exported to a split encoder/decoder pair -- ``<save-root>/<name>`` (full SAM weights, image
  encoder remapped) and ``<save-root>/<name>_decoder`` (AIS decoder) -- matching the pretrained
  ``vit_*_lm`` / ``vit_*_lm_decoder`` layout, so the final model loads directly in
  ``run_automatic_instance_segmentation`` (the ``_decoder`` file is auto-discovered).

Example (Kaggle, 2x T4)::

    !python candida_finetune_all.py \
        --raw-dir /kaggle/working/preprocessed/raw \
        --label-dir /kaggle/working/preprocessed/labels \
        --encoder /kaggle/input/.../vit_b_lm \
        --decoder /kaggle/input/.../vit_b_lm_decoder \
        --save-root /kaggle/working/final_model \
        --model-type vit_b_lm \
        --patch-shape 512 512 \
        --batch-size 2 \
        --lr 1e-5
"""

import os
import csv
import argparse
import statistics
from collections import OrderedDict

import numpy as np
import torch

import torch_em
from torch_em.data.sampler import MinInstanceSampler
from torch_em.multi_gpu_training import train_multi_gpu
from torch_em.util import load_image

from micro_sam.training import default_sam_dataset
from micro_sam.training.util import require_8bit

# Reuse the exact CV building blocks (model factory, per-rank logger, fixed-crop val dataset,
# train augmentation, data discovery and the before/after AIS comparison helpers) so there is no
# duplicated training / inference code.
from candida_multigpu_ais import (
    build_unetr_model,
    RankTensorboardLogger,
    SyncedValTrainer,
    fixed_crop_val_dataset,
    build_train_transform,
    _discover_pairs,
    _to_channels_last,
    _load_predictor_and_segmenter,
    _run_ais,
    _save_comparison_figure,
    _safe_load_checkpoint,
)


def _no_augmentation(raw, labels):
    """Identity transform: disable all augmentation for validation.

    ``default_sam_dataset`` substitutes the default (flip + 90-degree rotation) pipeline when
    ``transform is None``, so the fixed validation patch would still be randomly rotated/flipped
    unless an explicit no-op is supplied. Kept at module level (not a lambda) so it survives the
    ``mp.spawn`` pickle on the DDP worker path.
    """
    return raw, labels


def _export_final_model(best_checkpoint, encoder_out, decoder_out, model_type, encoder):
    """Export the DDP ``best.pt`` into a split encoder + decoder pair (the pretrained layout).

    Writes two standalone files matching the format of the pretrained ``vit_*_lm`` weights:
      * ``encoder_out``  -- the full SAM ``state_dict`` (``image_encoder.*`` / ``prompt_encoder.*`` /
        ``mask_decoder.*``), drop-in for ``--encoder`` / micro-sam ``checkpoint=``.
      * ``decoder_out``  -- the AIS decoder ``state_dict`` (``decoder.*`` / ``deconv*`` / ...),
        drop-in for ``--decoder``. Naming it ``<encoder_out>_decoder`` lets ``get_sam_model``
        auto-discover it.

    ``train_multi_gpu`` trains a ``DistributedDataParallel``-wrapped model, so ``best.pt``'s
    ``model_state`` keys are prefixed ``module.`` (e.g. ``module.encoder.pos_embed``).
    ``export_instance_segmentation_model`` filters for keys starting with ``encoder`` and would
    miss them, so the prefix is stripped into a temporary checkpoint first. The canonical export
    (which remaps ``image_encoder.* <- encoder.*``) is run to a temp combined file, then split so
    the remap logic stays entirely in micro-sam. Returns ``(encoder_out, decoder_out)``, or None
    if ``best.pt`` is missing.
    """
    import micro_sam.training as sam_training

    if not os.path.exists(best_checkpoint):
        print(f"[finetune-all] export skipped: no checkpoint at {best_checkpoint}")
        return None

    state = _safe_load_checkpoint(best_checkpoint)
    state.pop("init", None)  # drop the un-portable trainer-init blob (datasets + transforms) before re-save
    model_state = state.get("model_state", None)
    if model_state is not None and all(k.startswith("module.") for k in model_state):
        state["model_state"] = OrderedDict(
            (k[len("module."):], v) for k, v in model_state.items()
        )
    # Always re-save a cleaned copy (DDP prefix stripped, init dropped) so the export reads a portable
    # checkpoint regardless of how best.pt was produced.
    trained_path = best_checkpoint + ".no_ddp.tmp"
    torch.save(state, trained_path)
    print("[finetune-all] wrote portable checkpoint (DDP prefix stripped, init dropped) before export")

    # Export to a temp combined file via the canonical helper, then split it into two standalone
    # files. This keeps the image-encoder remapping in micro-sam (no reimplementation here).
    combined_tmp = encoder_out + ".combined.tmp"
    sam_training.export_instance_segmentation_model(
        trained_model_path=trained_path, output_path=combined_tmp,
        model_type=model_type, initial_checkpoint_path=encoder,
    )
    combined = _safe_load_checkpoint(combined_tmp)
    out_dir = os.path.dirname(encoder_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(OrderedDict(combined["model_state"]), encoder_out)
    torch.save(OrderedDict(combined["decoder_state"]), decoder_out)
    for tmp in (trained_path, combined_tmp):
        if tmp != best_checkpoint and os.path.exists(tmp):
            os.remove(tmp)
    print(f"[finetune-all] exported split model -> encoder: {encoder_out}  decoder: {decoder_out}")
    return encoder_out, decoder_out


def _val_patch_losses(args, raw_paths, label_paths, best_ckpt, device):
    """Per-image validation-patch loss for the fine-tuned model.

    The all-data validation set is one fixed, cell-rich patch per image (``fixed_crop_val_dataset``),
    so there are exactly N validation patches for N images. The trainer only ever reports the *mean*
    over them, so this recomputes ``DiceBasedDistanceLoss`` for each image's patch individually --
    exactly as ``DefaultTrainer._validate_impl`` does (``loss(model(x), y)`` on the 3-channel distance
    output) -- using the weights from ``best.pt`` (the checkpoint that drove selection). The mean of
    these per-patch losses therefore reproduces the saved ``best_metric``. Runs in fp32 (no autocast)
    to avoid the fp16 decoder-NaN issue; values are within rounding of the AMP training metric.
    Returns ``{stem: loss}`` (empty if ``best.pt`` is missing).
    """
    if not os.path.exists(best_ckpt):
        print(f"[finetune-all] val-patch loss skipped: no checkpoint at {best_ckpt}")
        return {}

    # Rebuild the UNETR and load the fine-tuned weights (strip the DDP 'module.' prefix).
    model = build_unetr_model(
        model_type=args.model_type, checkpoint_path=args.encoder, decoder_path=args.decoder,
        freeze=args.freeze, strict_decoder_loading=not args.flexible_decoder_loading,
    )
    state = _safe_load_checkpoint(best_ckpt)["model_state"]
    if all(k.startswith("module.") for k in state):
        state = OrderedDict((k[len("module."):], v) for k, v in state.items())
    model.load_state_dict(state)
    model = model.to(device).eval()

    loss_fn = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)
    shared_ds_kwargs = dict(
        raw_key=None, label_key=None, patch_shape=tuple(args.patch_shape),
        with_segmentation_decoder=True, train_instance_segmentation_only=True,
        with_channels=True, raw_transform=require_8bit,
    )

    losses = {}
    for rp, lp in zip(raw_paths, label_paths):
        stem = os.path.splitext(os.path.basename(rp))[0]
        # One image -> the same single fixed validation patch the trainer used (augmentation off).
        ds = fixed_crop_val_dataset(
            raw_paths=[rp], label_paths=[lp], is_train=False,
            transform=_no_augmentation, **shared_ds_kwargs,
        )
        x, y = ds[0]
        x = torch.as_tensor(x).float().unsqueeze(0).to(device)
        y = torch.as_tensor(y).float().unsqueeze(0).to(device)
        with torch.no_grad():
            loss = loss_fn(model(x), y)
        losses[stem] = float(loss.item())
        print(f"[finetune-all] val-patch loss {stem}: {losses[stem]:.6f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return losses


def compare_all_images(
    raw_paths, label_paths, model_type, encoder, decoder, finetuned_encoder, finetuned_decoder, figure_dir,
    device=None, tile_shape=(512, 512), halo=(64, 64),
):
    """Before/after whole-image AIS on every (training) image: 2x6 figures + mSA.

    NOTE: every image here was used for training, so this mSA is a *fit / sanity* check on the
    training data, **not** a generalization estimate (the cross-validation script provides that).

    Loads the original (``encoder``/``decoder``) and fine-tuned (the exported
    ``finetuned_encoder``/``finetuned_decoder`` pair) models **once each** -- not per image -- then
    for every image runs tiled whole-image AIS with both, scores each against the ground truth with
    mean segmentation accuracy, and writes ``comparison_<stem>.png``.
    Returns a list of ``{"stem", "msa_original", "msa_finetuned"}`` dicts.
    """
    from micro_sam.util import get_device
    from elf.evaluation import mean_segmentation_accuracy

    device = get_device(device)
    is_tiled = tile_shape is not None
    os.makedirs(figure_dir, exist_ok=True)

    # Load each model once; the segmenter is re-initialized per image inside _run_ais.
    orig_predictor, orig_segmenter = _load_predictor_and_segmenter(
        model_type, encoder, decoder, device, is_tiled,
    )
    ft_predictor, ft_segmenter = _load_predictor_and_segmenter(
        model_type, finetuned_encoder, finetuned_decoder, device, is_tiled,
    )

    per_image = []
    for rp, lp in zip(raw_paths, label_paths):
        stem = os.path.splitext(os.path.basename(str(rp)))[0]
        raw_image = _to_channels_last(np.asarray(load_image(rp)))
        gt_labels = np.asarray(load_image(lp))

        results, msa = {}, {}
        results["original"] = _run_ais(orig_predictor, orig_segmenter, raw_image, tile_shape, halo)
        msa["original"] = float(mean_segmentation_accuracy(results["original"]["instances"], gt_labels))
        results["finetuned"] = _run_ais(ft_predictor, ft_segmenter, raw_image, tile_shape, halo)
        msa["finetuned"] = float(mean_segmentation_accuracy(results["finetuned"]["instances"], gt_labels))

        out_path = os.path.join(figure_dir, f"comparison_{stem}.png")
        # "(train-set)" in the title flags that this is an optimistic fit check, not generalization.
        _save_comparison_figure(raw_image, gt_labels, results, msa, out_path, model_type, f"{stem} (train-set)")
        print(f"[finetune-all] {stem}: mSA original={msa['original']:.4f} -> finetuned={msa['finetuned']:.4f}")
        per_image.append({"stem": stem, "msa_original": msa["original"], "msa_finetuned": msa["finetuned"]})

    del orig_predictor, orig_segmenter, ft_predictor, ft_segmenter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_image


def _write_final_results(args, save_root, raw_paths, val_losses, per_image_msa):
    """Persist one row per image (val-patch loss + before/after mSA) and print the aggregate summary.

    Writes ``<save-root>/<name>_final_results.csv`` with columns
    ``image, val_patch_loss, msa_original, msa_finetuned``. ``val_patch_loss`` is the fine-tuned
    model's ``DiceBasedDistanceLoss`` on that image's fixed validation patch; the mSA columns are the
    before/after whole-image scores (computed on TRAINING images here -- a fit/sanity check, not a
    generalization estimate). Numeric fields use 6-decimal precision; missing values (e.g. mSA when
    ``--no-comparison`` is set, or anything when no ``best.pt`` was produced) are left blank.
    """
    fieldnames = ["image", "val_patch_loss", "msa_original", "msa_finetuned"]

    def _fmt(v):
        return "" if v is None else f"{v:.6f}"

    msa_by_stem = {d["stem"]: d for d in per_image_msa}
    rows = []
    for rp in raw_paths:
        stem = os.path.splitext(os.path.basename(rp))[0]
        m = msa_by_stem.get(stem, {})
        rows.append({
            "image": stem,
            "val_patch_loss": _fmt(val_losses.get(stem)),
            "msa_original": _fmt(m.get("msa_original")),
            "msa_finetuned": _fmt(m.get("msa_finetuned")),
        })

    csv_path = os.path.join(save_root, f"{args.name}_final_results.csv")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[finetune-all] wrote per-image results -> {csv_path}")

    # Printed aggregate summary (mirrors the CV script's style).
    vlosses = list(val_losses.values())
    o_msa = [d["msa_original"] for d in per_image_msa if d["msa_original"] is not None]
    f_msa = [d["msa_finetuned"] for d in per_image_msa if d["msa_finetuned"] is not None]
    print(f"{'=' * 78}\nFinal all-data summary ({len(raw_paths)} images)")
    if vlosses:
        print(f"  mean val-patch loss (lower is better) = {statistics.mean(vlosses):.6f}")
    if f_msa:
        o_mean = statistics.mean(o_msa) if o_msa else float("nan")
        f_mean = statistics.mean(f_msa)
        print(f"  train-set mSA (fit check, NOT generalization -- see CV for that): "
              f"original {o_mean:.4f} -> finetuned {f_mean:.4f}")
    print("=" * 78)


def run_training(args):
    """Run one DDP training session on all labeled images and export the final model."""
    raw_paths, label_paths = _discover_pairs(args.raw_dir, args.label_dir)
    print(f"Discovered {len(raw_paths)} image/label pairs -> training on ALL of them.")
    print(f"  images: {[os.path.basename(p) for p in raw_paths]}")

    sampler = MinInstanceSampler(2, min_size=25)
    shared_ds_kwargs = dict(
        raw_key=None,
        label_key=None,
        patch_shape=tuple(args.patch_shape),
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        with_channels=True,
        raw_transform=require_8bit,
        sampler=sampler,
    )

    # All images train (augmented); validation is a fixed cell-rich patch per image with
    # augmentation explicitly disabled (transform=_no_augmentation), so the val metric reflects
    # model state rather than the random crop / rotation drawn.
    train_dataset_kwargs = dict(
        raw_paths=raw_paths, label_paths=label_paths,
        is_train=True, transform=build_train_transform(), **shared_ds_kwargs,
    )
    val_dataset_kwargs = dict(
        raw_paths=raw_paths, label_paths=label_paths,
        is_train=False, transform=_no_augmentation, **shared_ds_kwargs,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        # Under DDP (mp.spawn) workers re-import the whole stack on creation; keep them alive
        # across epochs so that import cost is paid once, not per epoch.
        persistent_workers=args.num_workers > 0,
    )

    loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)

    print(f"\n{'=' * 70}\nFinal all-data fine-tune: '{args.name}'\n"
          f"  {len(raw_paths)} train images, validation = fixed crops of the same images\n{'=' * 70}")

    train_multi_gpu(
        model_callable=build_unetr_model,
        model_kwargs=dict(
            model_type=args.model_type,
            checkpoint_path=args.encoder,
            decoder_path=args.decoder,
            freeze=args.freeze,
            strict_decoder_loading=not args.flexible_decoder_loading,
        ),
        train_dataset_callable=default_sam_dataset,
        train_dataset_kwargs=train_dataset_kwargs,
        val_dataset_callable=fixed_crop_val_dataset,  # fixed cell-rich patch per image
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=int(args.iterations),
        # Every trainable parameter receives a gradient each step (unless freezing), so
        # find_unused_parameters is only enabled as a conservative guard when freezing.
        find_unused_parameters=args.freeze is not None,
        optimizer_callable=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=args.lr),
        lr_scheduler_callable=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs=dict(mode="min", factor=0.9, patience=3),
        # trainer params (forwarded to DefaultTrainer via **kwargs). SyncedValTrainer all-reduces
        # the per-rank validation metric so early stopping / checkpoint selection / LR scheduling
        # fire identically on every rank (otherwise ranks desync and dead-lock in the gradient
        # all-reduce). See candida_multigpu_ais.SyncedValTrainer.
        trainer_callable=SyncedValTrainer,
        logger=RankTensorboardLogger,  # each rank -> logs/<name>/rank<K>/ (separate TB runs)
        logger_kwargs=dict(spike_image_threshold=args.spike_image_threshold),
        name=args.name,
        save_root=args.save_root,
        loss=loss,
        metric=loss,
        early_stopping=args.early_stopping,
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False,
    )

    # Report the best metric and export the final model (runs in the main process, post-DDP).
    save_root = "" if args.save_root is None else args.save_root
    best_ckpt = os.path.join(save_root, "checkpoints", args.name, "best.pt")
    if os.path.exists(best_ckpt):
        best_metric = _safe_load_checkpoint(best_ckpt).get("best_metric")
        print(f"\nBest validation metric (fixed train crops) = {best_metric:.6f}")
    else:
        print(f"\nNo best.pt found at {best_ckpt}")

    # Default split-export paths: encoder at <save-root>/<name>, decoder alongside as <name>_decoder
    # (the pretrained vit_*_lm / vit_*_lm_decoder layout; the _decoder suffix is auto-discovered).
    encoder_out = args.export_path or os.path.join(save_root, args.name)
    decoder_out = encoder_out + "_decoder"

    if not args.no_export:
        _export_final_model(best_ckpt, encoder_out, decoder_out, args.model_type, args.encoder)

    # Per-image diagnostics (main process, post-DDP): the fine-tuned model's loss on each image's
    # fixed validation patch, plus an optional before/after whole-image AIS comparison (figure + mSA)
    # on every image. NOTE: every image was used for training, so the mSA here is a fit / sanity check
    # on the training data -- NOT a generalization estimate (the CV script provides that).
    from micro_sam.util import get_device
    device = get_device(None)

    val_losses = _val_patch_losses(args, raw_paths, label_paths, best_ckpt, device)

    per_image_msa = []
    if not args.no_comparison and os.path.exists(best_ckpt):
        figure_dir = os.path.join(save_root, "logs", args.name)
        os.makedirs(figure_dir, exist_ok=True)
        # The comparison needs an AIS-ready fine-tuned model. Reuse the exported split pair if
        # present; otherwise export a temporary pair just for the comparison and delete it after.
        tmp_pair = None
        if os.path.exists(encoder_out) and os.path.exists(decoder_out):
            finetuned_encoder, finetuned_decoder = encoder_out, decoder_out
        else:
            tmp_encoder = os.path.join(figure_dir, "_finetuned")
            tmp_decoder = tmp_encoder + "_decoder"
            exported = _export_final_model(best_ckpt, tmp_encoder, tmp_decoder, args.model_type, args.encoder)
            if exported:
                finetuned_encoder, finetuned_decoder = exported
                tmp_pair = exported
            else:
                finetuned_encoder = finetuned_decoder = None
        if finetuned_encoder and os.path.exists(finetuned_encoder):
            per_image_msa = compare_all_images(
                raw_paths, label_paths, args.model_type, args.encoder, args.decoder,
                finetuned_encoder, finetuned_decoder, figure_dir, device=device,
                tile_shape=tuple(args.comparison_tile_shape), halo=tuple(args.comparison_halo),
            )
        if tmp_pair:
            for f in tmp_pair:
                if os.path.exists(f):
                    os.remove(f)

    _write_final_results(args, save_root, raw_paths, val_losses, per_image_msa)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU (DDP) final fine-tune of SAM encoder + AIS decoder on ALL labeled data."
    )
    parser.add_argument("--raw-dir", required=True, help="Directory of pre-processed raw .tif images (one per file).")
    parser.add_argument("--label-dir", required=True, help="Directory of label .tif images (one per file).")
    parser.add_argument("--encoder", default=None, help="Path to the SAM encoder checkpoint (vit_*_lm).")
    parser.add_argument("--decoder", default=None, help="Path to the AIS decoder checkpoint (vit_*_lm_decoder).")
    parser.add_argument("--save-root", default=None, help="Root dir for checkpoints/ and logs/.")
    parser.add_argument("--name", default="microsam_candida_final", help="Run name.")
    parser.add_argument("--model-type", default="vit_b_lm", help="SAM model type, e.g. vit_b_lm / vit_l_lm.")
    parser.add_argument("--patch-shape", type=int, nargs=2, default=[512, 512], help="Training patch shape H W.")
    parser.add_argument("--iterations", type=int, default=int(1e5), help="Max training iterations.")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Early-stopping patience in epochs (on the fixed-crop val metric). "
                             "Use a large value or 0/None to disable.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per process.")
    parser.add_argument("--spike-image-threshold", type=float, default=1.0,
                        help="Also log input/target/prediction images to a *_spike/ tag whenever the "
                             "train/val loss exceeds this value (DiceBasedDistanceLoss ranges ~0-3).")
    parser.add_argument("--freeze", type=str, nargs="+", default=None,
                        help="Model parts to freeze (e.g. image_encoder). Default: nothing frozen.")
    parser.add_argument("--flexible-decoder-loading", action="store_true",
                        help="Allow loading a decoder with mismatched output channels (reinitializes them).")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip exporting best.pt to the split encoder/decoder pair after training.")
    parser.add_argument("--export-path", default=None,
                        help="Encoder output path (vit_*_lm format); the decoder is written alongside "
                             "as <path>_decoder. Default: <save-root>/<name> (+ <name>_decoder).")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip the before/after AIS figure + mSA scoring on all (training) images. "
                             "The per-image validation-patch loss is still written.")
    parser.add_argument("--comparison-tile-shape", type=int, nargs=2, default=[512, 512],
                        help="Tile shape for the whole-image before/after AIS comparison inference.")
    parser.add_argument("--comparison-halo", type=int, nargs=2, default=[64, 64],
                        help="Per-tile overlap (halo) for the whole-image before/after AIS comparison inference.")
    args = parser.parse_args()

    # early_stopping=0 -> disable (DefaultTrainer treats None as 'no early stopping').
    if args.early_stopping is not None and args.early_stopping <= 0:
        args.early_stopping = None

    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device visible. This script requires at least one GPU.")
    print(f"Visible GPUs: {torch.cuda.device_count()} "
          f"({', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))})")

    run_training(args)


if __name__ == "__main__":
    main()
