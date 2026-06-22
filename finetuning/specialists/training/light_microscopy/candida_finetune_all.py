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
  is exported to an AIS-ready ``.pth`` (image encoder remapped + ``decoder_state``), so the
  final model loads directly in ``run_automatic_instance_segmentation`` without manual fixing.

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
import argparse
from collections import OrderedDict

import torch

import torch_em
from torch_em.data.sampler import MinInstanceSampler
from torch_em.multi_gpu_training import train_multi_gpu

from micro_sam.training import default_sam_dataset
from micro_sam.training.util import require_8bit

# Reuse the exact CV building blocks (model factory, per-rank logger, fixed-crop val dataset,
# train augmentation and data discovery) so there is no duplicated training code.
from candida_multigpu_ais import (
    build_unetr_model,
    RankTensorboardLogger,
    SyncedValTrainer,
    fixed_crop_val_dataset,
    build_train_transform,
    _discover_pairs,
)


def _no_augmentation(raw, labels):
    """Identity transform: disable all augmentation for validation.

    ``default_sam_dataset`` substitutes the default (flip + 90-degree rotation) pipeline when
    ``transform is None``, so the fixed validation patch would still be randomly rotated/flipped
    unless an explicit no-op is supplied. Kept at module level (not a lambda) so it survives the
    ``mp.spawn`` pickle on the DDP worker path.
    """
    return raw, labels


def _export_final_model(best_checkpoint, output_path, model_type, encoder):
    """Export the DDP ``best.pt`` into an AIS-ready model (image encoder remapped + decoder_state).

    ``train_multi_gpu`` trains a ``DistributedDataParallel``-wrapped model, so ``best.pt``'s
    ``model_state`` keys are prefixed ``module.`` (e.g. ``module.encoder.pos_embed``).
    ``export_instance_segmentation_model`` filters for keys starting with ``encoder`` and would
    miss them, so the prefix is stripped into a temporary checkpoint first. Returns the export
    path, or None if ``best.pt`` is missing.
    """
    import micro_sam.training as sam_training

    if not os.path.exists(best_checkpoint):
        print(f"[finetune-all] export skipped: no checkpoint at {best_checkpoint}")
        return None

    state = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model_state = state.get("model_state", None)
    if model_state is not None and all(k.startswith("module.") for k in model_state):
        state["model_state"] = OrderedDict(
            (k[len("module."):], v) for k, v in model_state.items()
        )
        trained_path = best_checkpoint + ".no_ddp.tmp"
        torch.save(state, trained_path)
        print("[finetune-all] stripped DDP 'module.' prefix before export")
    else:
        trained_path = best_checkpoint  # single-GPU / already-stripped checkpoint

    sam_training.export_instance_segmentation_model(
        trained_model_path=trained_path, output_path=output_path,
        model_type=model_type, initial_checkpoint_path=encoder,
    )
    if trained_path != best_checkpoint and os.path.exists(trained_path):
        os.remove(trained_path)
    print(f"[finetune-all] exported AIS-ready model -> {output_path}")
    return output_path


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
        best_metric = torch.load(best_ckpt, map_location="cpu", weights_only=False).get("best_metric")
        print(f"\nBest validation metric (fixed train crops) = {best_metric:.6f}")
    else:
        print(f"\nNo best.pt found at {best_ckpt}")

    if not args.no_export:
        export_path = args.export_path or os.path.join(save_root, f"{args.name}_ais.pth")
        _export_final_model(best_ckpt, export_path, args.model_type, args.encoder)


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
                        help="Skip exporting best.pt to an AIS-ready .pth after training.")
    parser.add_argument("--export-path", default=None,
                        help="Where to write the exported AIS model. Default: <save-root>/<name>_ais.pth.")
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
