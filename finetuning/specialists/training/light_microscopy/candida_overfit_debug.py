"""Single-image overfit sanity check for SAM encoder + AIS decoder fine-tuning.

This is a debugging tool, not a training script. It trains the UNETR (SAM encoder + AIS
decoder) on a *single* image with augmentation disabled and validation == train, then checks
whether the loss collapses toward ~0. If the model cannot memorize one image, the bug is in the
data / model / loss wiring rather than in generalization, and there is no point debugging the
full training run until this passes.

It deliberately reuses ``torch_em.trainer.DefaultTrainer`` -- the exact trainer the real
multi-GPU run uses (via ``train_multi_gpu``). By default it runs on a single GPU with no DDP, so
this probe exercises the real per-rank training step (same forward, loss application and logging)
and the TensorBoard output has the identical schema to the cross-validation runs
(``train/loss``, ``validation/loss``, ``validation/metric`` plus raw/target/prediction image
grids).

Pass ``--multi-gpu`` to instead distribute the probe across all local GPUs (e.g. Kaggle's
2x T4) via the same ``train_multi_gpu`` path the cross-validation uses. This reuses
``build_unetr_model`` from ``candida_multigpu_ais``, so no model code is duplicated. Both ranks
overfit the *same* single image (validation == train); ``DistributedSampler`` just splits the
``n_samples`` indices across them, giving an effective batch of one-per-GPU. This roughly halves
the wall-clock to memorization when a single T4 is already VRAM-bound at batch size 1.

The single-GPU default can be called directly from a notebook cell::

    from candida_overfit_debug import overfit_single_image
    result = overfit_single_image(
        raw_path="/kaggle/working/preprocessed/raw/img_000.tif",
        label_path="/kaggle/working/preprocessed/labels/img_000.tif",
        encoder="/kaggle/input/.../vit_b_lm",
        decoder="/kaggle/input/.../vit_b_lm_decoder",
        model_type="vit_b_lm",
        n_iterations=300,
        save_root="/kaggle/working/overfit_logs",
    )

The run "passes" if ``validation/metric`` collapses toward 0 (watch the curve in TensorBoard).
``DiceBasedDistanceLoss`` sums three per-channel Dice losses, so a perfectly memorized image
trends toward 0.
"""

import os
from typing import Optional, Union, Dict

import numpy as np
import torch
import torch.utils.data

import torch_em
from torch_em.data.sampler import MinInstanceSampler
from torch_em.multi_gpu_training import train_multi_gpu
from torch_em.util import load_image

from micro_sam.util import get_device
from micro_sam.training import default_sam_loader, default_sam_dataset
from micro_sam.training.util import get_trainable_sam_model, require_8bit
from micro_sam.instance_segmentation import get_unetr


def _build_model(model_type, encoder, decoder, device, strict_decoder_loading=True):
    """Same SAM-encoder + AIS-decoder UNETR that train_instance_segmentation builds."""
    sam_model, state = get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=encoder,
        return_state=True,
        freeze=None,  # train the full model so it can fully memorize the image
        decoder_path=decoder,
    )
    model = get_unetr(
        image_encoder=sam_model.sam.image_encoder,
        decoder_state=state.get("decoder_state", None),
        device=device,
        flexible_load_checkpoint=not strict_decoder_loading,
    )
    return model


def _no_augmentation(raw, labels):
    """Identity transform: disable all geometric/photometric augmentation.

    ``default_sam_dataset`` substitutes the default (flip + 90-degree rotation) pipeline when
    ``transform is None``, so an explicit no-op is required to truly disable augmentation. Kept
    at module level (not a lambda) so it survives the ``mp.spawn`` pickle on the DDP/worker paths.
    """
    return raw, labels


def _dataset_kwargs(raw_path, label_path, patch_shape, n_samples, is_train):
    """Shared default_sam_dataset kwargs for the single training image (no augmentation).

    Used both to build the single-GPU loader and as the train/val dataset kwargs for the
    multi-GPU (DDP) path, so the two paths see an identical dataset definition.
    """
    return dict(
        raw_paths=[str(raw_path)],
        label_paths=[str(label_path)],
        raw_key=None,
        label_key=None,
        patch_shape=tuple(patch_shape),
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        with_channels=True,
        is_train=is_train,
        raw_transform=require_8bit,
        # Validation must see the exact fixed crop (no flips/rotations) so its loss reflects model
        # state. transform=None would make default_sam_dataset substitute the default augmentation
        # pipeline, so the val path uses an explicit no-op; train keeps the default augmentations.
        transform=None if is_train else _no_augmentation,
        sampler=MinInstanceSampler(2, min_size=25),
        n_samples=n_samples,
    )


def _single_image_loader(raw_path, label_path, patch_shape, n_samples, is_train, num_workers=0):
    """Single-image AIS loader with augmentation disabled (single-GPU path)."""
    return default_sam_loader(
        **_dataset_kwargs(raw_path, label_path, patch_shape, n_samples, is_train),
        batch_size=1,
        num_workers=num_workers,
        # Keep workers alive across epochs; otherwise short (single-image) epochs respawn them
        # every epoch and each spawn re-imports the whole stack (num_workers>0 only).
        persistent_workers=num_workers > 0,
        shuffle=False,
    )


# ---------------------------------------------------------------------------
# Before/after full-image AIS comparison (from saved checkpoints). Runs once in
# the main process after training -- independent of the DDP (mp.spawn) workers --
# loading the original model from the saved encoder/decoder weights and the
# overfitted model from the saved best.pt checkpoint.
# ---------------------------------------------------------------------------
def _to_channels_last(image: np.ndarray) -> np.ndarray:
    """Channels-first ``(C, H, W)`` -> ``(H, W, C)``; leave grayscale / channels-last as-is.

    ``micro_sam.util._to_image`` treats the *last* axis as channels, so the pre-processed
    channels-first raw tiff must be transposed before inference. Reuses the same
    ``shape[-1] > 16`` channel-first heuristic as ``_best_variance_patch``.
    """
    if image.ndim == 3 and image.shape[-1] > 16:  # (C, H, W): last axis is spatial -> channels first
        return np.ascontiguousarray(np.transpose(image, (1, 2, 0)))
    return np.ascontiguousarray(image)


def _strip_ddp_prefix(checkpoint_path, figure_dir):
    """Return a checkpoint whose ``model_state`` keys have no ``module.`` (DDP) prefix.

    A ``--multi-gpu`` overfit run trains a ``DistributedDataParallel``-wrapped model, so
    ``DefaultTrainer`` saves ``self.model.state_dict()`` with every key prefixed ``module.``
    (e.g. ``module.encoder.pos_embed``). ``export_instance_segmentation_model`` filters for keys
    starting with ``encoder`` and would miss them, raising ``KeyError: 'encoder.pos_embed'``.

    If the checkpoint carries the prefix, write a normalized copy into ``figure_dir`` (stripping the
    prefix from ``model_state`` only) and return its path. Single-GPU checkpoints have no prefix, so
    the original path is returned unchanged (no copy).
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state.get("model_state", None)
    if model_state is None or not all(k.startswith("module.") for k in model_state):
        return checkpoint_path
    from collections import OrderedDict
    state["model_state"] = OrderedDict(
        (k[len("module."):], v) for k, v in model_state.items()
    )
    normalized_path = os.path.join(figure_dir, "overfit_best_no_ddp.pt")
    torch.save(state, normalized_path)
    print(f"[overfit-comparison] stripped DDP 'module.' prefix from checkpoint -> {normalized_path}")
    return normalized_path


def _load_predictor_and_segmenter(model_type, checkpoint_path, decoder_path, device, is_tiled):
    """Build a SAM predictor + AIS segmenter from saved weights (mirrors get_predictor_and_segmenter).

    ``checkpoint_path`` is loaded by ``get_sam_model`` (a SAM checkpoint, or an exported AIS model
    that also carries a ``decoder_state``). ``decoder_path`` is an optional separate decoder
    ``state_dict`` file; when None the decoder is taken from the loaded checkpoint's
    ``decoder_state`` (the exported overfitted model, or a bundled model decoder).
    """
    from micro_sam import util
    from micro_sam.instance_segmentation import get_decoder, get_instance_segmentation_generator

    predictor, state = util.get_sam_model(
        model_type=model_type, checkpoint_path=checkpoint_path, device=device, return_state=True,
    )
    if decoder_path is not None:
        decoder_state = torch.load(decoder_path, map_location=device, weights_only=False)
    elif "decoder_state" in state:
        decoder_state = state["decoder_state"]
    else:
        raise RuntimeError(
            f"No decoder weights: checkpoint '{checkpoint_path}' has no 'decoder_state' and no "
            "decoder_path was given."
        )
    decoder = get_decoder(predictor.model.image_encoder, decoder_state, device)
    segmenter = get_instance_segmentation_generator(predictor=predictor, is_tiled=is_tiled, decoder=decoder)
    return predictor, segmenter


def _run_ais(predictor, segmenter, image, tile_shape, halo):
    """Run whole-image AIS and return ``(instances, foreground)``."""
    from micro_sam import util

    is_tiled = tile_shape is not None
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor, input_=image, ndim=2, tile_shape=tile_shape, halo=halo,
    )
    init_kwargs = dict(image=image, image_embeddings=image_embeddings)
    # output_mode="instance_segmentation" returns the label image directly; None routes through
    # _to_masks, which this micro_sam version rejects.
    generate_kwargs = {"output_mode": "instance_segmentation"}
    if is_tiled:
        init_kwargs["batch_size"] = 1
        generate_kwargs.update(tile_shape=tile_shape, halo=halo)
    segmenter.initialize(**init_kwargs)
    instances = segmenter.generate(**generate_kwargs)
    foreground = segmenter.get_state()["foreground"]
    return instances, foreground


def _save_comparison_figure(raw_image, gt_labels, results, out_path, model_type):
    """2x4 grid: rows = (Original, Overfitted); cols = Raw, Ground truth, Instances, Foreground."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no interactive display in the training subprocess
    import matplotlib.pyplot as plt
    from torch_em.util.util import get_random_colors

    disp_raw = raw_image if raw_image.ndim == 2 else raw_image[..., :3]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for r, key in enumerate(("original", "overfitted")):
        instances, foreground = results[key]
        axes[r, 0].imshow(disp_raw, cmap="gray" if disp_raw.ndim == 2 else None)
        axes[r, 0].set_ylabel(key.capitalize(), fontsize=14)
        axes[r, 0].set_title("Raw" if r == 0 else "")
        axes[r, 1].imshow(gt_labels, cmap=get_random_colors(gt_labels), interpolation="nearest")
        axes[r, 1].set_title(f"Ground truth (n={int(gt_labels.max())})" if r == 0 else "")
        axes[r, 2].imshow(instances, cmap=get_random_colors(instances), interpolation="nearest")
        axes[r, 2].set_title(f"Instances (n={int(instances.max())})")
        axes[r, 3].imshow(foreground, cmap="viridis")
        axes[r, 3].set_title("Foreground prob")
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle(f"Overfit before/after - {model_type}", fontsize=16)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compare_full_image(
    raw_path, label_path, model_type, encoder, decoder, best_checkpoint, figure_dir,
    device=None, tile_shape=(512, 512), halo=(64, 64),
):
    """Segment the whole image with the original and overfitted model and save a before/after figure.

    Disk-based and run in the main process after training (independent of the DDP workers):
    the **original** model is loaded from the saved ``encoder`` / ``decoder`` weights, and the
    **overfitted** model from the saved ``best.pt``, which is first exported to an AIS-ready
    checkpoint (image encoder remapped + ``decoder_state``) via
    ``export_instance_segmentation_model``. Returns the saved figure path, or None if skipped.
    """
    import micro_sam.training as sam_training
    from micro_sam.util import get_device

    if not os.path.exists(best_checkpoint):
        print(f"[overfit-comparison] skipped: no checkpoint at {best_checkpoint}")
        return None

    # best.pt embeds the validation dataset, whose transform is this module's _no_augmentation.
    # The overfit ran as a script (__main__), so it was pickled as __main__._no_augmentation;
    # register it on __main__ here so torch.load resolves it when run from another __main__
    # (e.g. a notebook). This is the only script-local symbol the checkpoint references.
    import __main__
    if not hasattr(__main__, "_no_augmentation"):
        __main__._no_augmentation = _no_augmentation

    device = get_device(device)
    is_tiled = tile_shape is not None
    raw_image = _to_channels_last(np.asarray(load_image(raw_path)))
    gt_labels = np.asarray(load_image(label_path))

    # Export the overfit checkpoint into an AIS-ready model (image encoder remapped + decoder_state).
    # A --multi-gpu run saved a DDP-wrapped model, so first strip any 'module.' prefix that would
    # otherwise make export_instance_segmentation_model's encoder filter miss every weight.
    os.makedirs(figure_dir, exist_ok=True)
    normalized_checkpoint = _strip_ddp_prefix(best_checkpoint, figure_dir)
    export_path = os.path.join(figure_dir, "overfit_export.pth")
    sam_training.export_instance_segmentation_model(
        trained_model_path=normalized_checkpoint, output_path=export_path,
        model_type=model_type, initial_checkpoint_path=encoder,
    )

    results = {}
    runs = (
        ("original", encoder, decoder),     # saved pretrained weights (decoder is a separate file)
        ("overfitted", export_path, None),  # exported best.pt (decoder lives inside the checkpoint)
    )
    for key, checkpoint_path, decoder_path in runs:
        predictor, segmenter = _load_predictor_and_segmenter(
            model_type, checkpoint_path, decoder_path, device, is_tiled,
        )
        results[key] = _run_ais(predictor, segmenter, raw_image, tile_shape, halo)
        del predictor, segmenter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(export_path):
        os.remove(export_path)  # ~400 MB; the figure is what we keep
    if normalized_checkpoint != best_checkpoint and os.path.exists(normalized_checkpoint):
        os.remove(normalized_checkpoint)  # temp DDP-stripped copy; best.pt stays untouched

    out_path = os.path.join(figure_dir, "overfit_comparison.png")
    _save_comparison_figure(raw_image, gt_labels, results, out_path, model_type)
    print(f"[overfit-comparison] saved before/after figure to: {out_path}")
    return out_path


def overfit_single_image(
    raw_path: Union[str, os.PathLike],
    label_path: Union[str, os.PathLike],
    encoder: Optional[str] = None,
    decoder: Optional[str] = None,
    model_type: str = "vit_b_lm",
    patch_shape=(512, 512),
    n_iterations: int = 300,
    lr: float = 1e-4,
    iters_per_epoch: int = 25,
    log_image_interval: int = 10,
    save_root: Optional[str] = None,
    name: Optional[str] = None,
    pass_threshold: float = 0.1,
    device: Optional[Union[str, torch.device]] = None,
    multi_gpu: bool = False,
    mixed_precision: bool = True,
    num_workers: int = 0,
    spike_image_threshold: float = 1.0,
    full_image_comparison: bool = True,
    comparison_dir: Optional[str] = None,
    comparison_tile_shape=(512, 512),
    comparison_halo=(64, 64),
) -> Dict:
    """Train on one image (no augmentation, validation == train) via DefaultTrainer.

    Args:
        raw_path: Path to a single pre-processed raw .tif (channels-first).
        label_path: Path to the matching instance-label .tif.
        encoder: Path to the SAM encoder checkpoint (vit_*_lm). None uses default weights.
        decoder: Path to the AIS decoder checkpoint (vit_*_lm_decoder). None uses default.
        model_type: SAM model type, e.g. vit_b_lm / vit_l_lm.
        patch_shape: Training patch shape (H, W).
        n_iterations: Total number of optimization steps.
        lr: Learning rate. A larger LR than the real training (1e-4) is used to overfit fast.
        iters_per_epoch: Iterations between validation passes (shorter = denser validation curve).
        log_image_interval: How often (in iterations) DefaultTrainer logs image grids to TensorBoard.
        save_root: Root for TensorBoard logs (``<save_root>/logs/<name>``) and checkpoints
            (``<save_root>/checkpoints/<name>``). If None, the current working directory is used.
        name: Run name. Defaults to ``overfit_<raw-stem>``.
        pass_threshold: ``validation/metric`` below this is reported as PASS. Heuristic only --
            the TensorBoard curve is the real evidence.
        device: Torch device (single-GPU path only). Defaults to the best available.
        multi_gpu: If True, distribute across all local GPUs via ``train_multi_gpu`` (DDP).
            Must be launched as a script (``mp.spawn`` re-imports the module), not called inline.
        mixed_precision: Use fp16 autocast. Safe on T4 (Tensor Cores) and lowers VRAM; set False
            for an fp32-pure probe.
        num_workers: DataLoader workers per process.
        spike_image_threshold: Also log input/target/prediction to a ``*_spike/`` tag whenever the
            train/val loss exceeds this value (helps diagnose loss spikes off the periodic interval).
        full_image_comparison: After training, segment the *whole* image with the original and the
            overfitted model (both from in-memory weights, no checkpoint reload) and save a
            before/after figure. Runs on rank 0 only, so it is safe under DDP.
        comparison_dir: Directory for the comparison figure, written as
            ``<comparison_dir>/overfit_comparison.png``. Defaults to this run's TensorBoard log
            dir (``<save_root>/logs/<name>``), so the figure sits next to the overfit logs.
        comparison_tile_shape: Tile shape for the whole-image AIS inference. Default ``(512, 512)``.
        comparison_halo: Per-tile overlap (halo) for stitching. Default ``(64, 64)``.

    Returns:
        Dict with best_metric, latest_metric, passed, name and log_dir.
    """
    # Reuse the CV script's logger (per-rank subfolders + loss-spike image capture) and the
    # fixed cell-rich validation patch, so the single-GPU and multi-GPU paths stay consistent.
    from candida_multigpu_ais import RankTensorboardLogger, fixed_crop_val_dataset

    if name is None:
        stem = os.path.splitext(os.path.basename(str(raw_path)))[0]
        name = f"overfit_{stem}"
    log_dir = os.path.join(save_root or ".", "logs", name)

    loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)
    n_val = max(1, iters_per_epoch // 5)

    mode = (f"multi-GPU DDP ({torch.cuda.device_count()} GPUs)" if multi_gpu else "single process")
    print(f"Overfit sanity check on:\n  raw:   {raw_path}\n  label: {label_path}\n"
          f"  mode: {mode}\n  mixed_precision: {mixed_precision}\n  tensorboard log_dir: {log_dir}")

    if multi_gpu:
        # Reuse the exact model factory and DDP wiring the cross-validation script uses.
        from candida_multigpu_ais import build_unetr_model

        # Both train and val datasets point at the same single image (validation == train);
        # DistributedSampler splits the n_samples indices across ranks -> effective batch = 1/GPU.
        # Validation uses a fixed cell-rich patch (fixed_crop_val_dataset) so its loss reflects
        # model state, not random crop selection.
        train_multi_gpu(
            model_callable=build_unetr_model,
            model_kwargs=dict(
                model_type=model_type, checkpoint_path=encoder, decoder_path=decoder,
                freeze=None, strict_decoder_loading=True,
            ),
            train_dataset_callable=default_sam_dataset,
            train_dataset_kwargs=_dataset_kwargs(raw_path, label_path, patch_shape, iters_per_epoch, True),
            val_dataset_callable=fixed_crop_val_dataset,
            val_dataset_kwargs=_dataset_kwargs(raw_path, label_path, patch_shape, n_val, False),
            # persistent_workers: under DDP (mp.spawn) workers re-import the whole stack on creation,
            # so keep them alive across epochs instead of respawning each epoch (num_workers>0 only).
            loader_kwargs=dict(batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
                               persistent_workers=num_workers > 0),
            iterations=n_iterations,
            # The full model is trained (freeze=None), so every parameter gets a gradient each
            # step; find_unused_parameters=True would only add per-iteration graph-traversal overhead.
            find_unused_parameters=False,
            optimizer_callable=torch.optim.AdamW,
            optimizer_kwargs=dict(lr=lr),
            # trainer params (forwarded to DefaultTrainer via **kwargs)
            trainer_callable=torch_em.trainer.DefaultTrainer,
            logger=RankTensorboardLogger,  # each rank -> logs/<name>/rank<K>/ (separate TB runs)
            logger_kwargs=dict(spike_image_threshold=spike_image_threshold),
            name=name,
            save_root=save_root,
            loss=loss,
            metric=loss,
            early_stopping=None,  # we want to watch it overfit, not stop early
            mixed_precision=mixed_precision,
            log_image_interval=log_image_interval,
            compile_model=False,
        )
    else:
        device = get_device(device)
        print(f"  device: {device}")

        # Training draws random crops of the single image; validation uses a fixed cell-rich
        # patch (fixed_crop_val_dataset) so its loss reflects model state, not crop selection.
        train_loader = _single_image_loader(raw_path, label_path, patch_shape, iters_per_epoch, True, num_workers)
        val_dataset = fixed_crop_val_dataset(
            **_dataset_kwargs(raw_path, label_path, patch_shape, n_val, False)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
            persistent_workers=num_workers > 0,  # avoid per-epoch worker respawn + re-import
        )

        model = _build_model(model_type, encoder, decoder, device)

        trainer = torch_em.trainer.DefaultTrainer(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss=loss,
            metric=loss,
            optimizer=torch.optim.AdamW(model.parameters(), lr=lr),
            device=device,
            mixed_precision=mixed_precision,
            log_image_interval=log_image_interval,
            logger=RankTensorboardLogger,  # spike-image capture (rank is None -> base log dir)
            logger_kwargs=dict(spike_image_threshold=spike_image_threshold),
            early_stopping=None,  # we want to watch it overfit, not stop early
            save_root=save_root,
            compile_model=False,
        )
        trainer.fit(iterations=n_iterations)

    # Read the best / latest validation metric back from the saved checkpoints.
    ckpt_dir = os.path.join(save_root or ".", "checkpoints", name)

    def _metric(fname):
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu", weights_only=False).get("best_metric")

    best_metric = _metric("best.pt")
    latest_metric = torch.load(os.path.join(ckpt_dir, "latest.pt"), map_location="cpu",
                               weights_only=False).get("current_metric") \
        if os.path.exists(os.path.join(ckpt_dir, "latest.pt")) else None

    passed = best_metric is not None and best_metric < pass_threshold
    print(f"\nbest validation/metric  = {best_metric}")
    print(f"latest validation/metric = {latest_metric}")
    print(f"-> {'PASS' if passed else 'SUSPECT'} (threshold {pass_threshold})")
    if not passed:
        print("  validation/metric did not collapse. Check data wiring (channels, label "
              "transform), model loading or learning rate before debugging the full run.")
    print(f"  Inspect the live curves and image grids in TensorBoard at: {log_dir}")

    # After training (and best.pt is on disk), segment the whole image with the original and
    # overfitted model and save a before/after figure. Runs here in the main process, so it is
    # independent of the DDP workers. A failure here must not invalidate a finished overfit run.
    if full_image_comparison:
        try:
            compare_full_image(
                raw_path=raw_path, label_path=label_path, model_type=model_type,
                encoder=encoder, decoder=decoder,
                best_checkpoint=os.path.join(ckpt_dir, "best.pt"),
                figure_dir=comparison_dir or log_dir,
                device=device if not multi_gpu else None,
                tile_shape=tuple(comparison_tile_shape), halo=tuple(comparison_halo),
            )
        except Exception as e:
            import traceback
            print(f"[overfit-comparison] skipped: {e}")
            traceback.print_exc()

    return {
        "best_metric": best_metric,
        "latest_metric": latest_metric,
        "passed": passed,
        "name": name,
        "log_dir": log_dir,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Single-image overfit sanity check for AIS fine-tuning.")
    parser.add_argument("--raw", required=True, help="Path to a single pre-processed raw .tif.")
    parser.add_argument("--label", required=True, help="Path to the matching label .tif.")
    parser.add_argument("--encoder", default=None, help="SAM encoder checkpoint path.")
    parser.add_argument("--decoder", default=None, help="AIS decoder checkpoint path.")
    parser.add_argument("--model-type", default="vit_b_lm")
    parser.add_argument("--patch-shape", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--pass-threshold", type=float, default=0.1,
                        help="Report PASS when best validation/metric drops below this. Heuristic.")
    parser.add_argument("--iters-per-epoch", type=int, default=25,
                        help="Iterations between validation passes.")
    parser.add_argument("--log-image-interval", type=int, default=10,
                        help="Log image grids to TensorBoard every N iterations.")
    parser.add_argument("--save-root", default=None,
                        help="Root for logs/<name> and checkpoints/<name>. Default: cwd.")
    parser.add_argument("--name", default=None, help="Run name. Default: overfit_<raw-stem>.")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Distribute across all local GPUs via DDP (train_multi_gpu). "
                             "Default: single GPU. Effective batch = 1 per GPU.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per process.")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True,
                        help="Use fp16 autocast (safe on T4, lowers VRAM). Use --no-mixed-precision for fp32.")
    parser.add_argument("--spike-image-threshold", type=float, default=1.0,
                        help="Also log input/target/prediction to a *_spike/ tag whenever the train/val "
                             "loss exceeds this value (DiceBasedDistanceLoss ranges ~0-3).")
    parser.add_argument("--full-image-comparison", action=argparse.BooleanOptionalAction, default=True,
                        help="After training, segment the whole image with the original and overfitted "
                             "model (in-memory) and save a before/after figure. --no-full-image-comparison to skip.")
    parser.add_argument("--comparison-dir", default=None,
                        help="Directory for the comparison figure (<dir>/overfit_comparison.png). "
                             "Default: the run's TensorBoard log dir (<save-root>/logs/<name>).")
    parser.add_argument("--comparison-tile-shape", type=int, nargs=2, default=[512, 512],
                        help="Tile shape for the whole-image AIS comparison inference.")
    parser.add_argument("--comparison-halo", type=int, nargs=2, default=[64, 64],
                        help="Per-tile overlap (halo) for the whole-image AIS comparison inference.")
    args = parser.parse_args()

    overfit_single_image(
        raw_path=args.raw, label_path=args.label,
        encoder=args.encoder, decoder=args.decoder,
        model_type=args.model_type, patch_shape=tuple(args.patch_shape),
        n_iterations=args.iterations, lr=args.lr, pass_threshold=args.pass_threshold,
        iters_per_epoch=args.iters_per_epoch, log_image_interval=args.log_image_interval,
        save_root=args.save_root, name=args.name,
        multi_gpu=args.multi_gpu, mixed_precision=args.mixed_precision, num_workers=args.num_workers,
        spike_image_threshold=args.spike_image_threshold,
        full_image_comparison=args.full_image_comparison, comparison_dir=args.comparison_dir,
        comparison_tile_shape=tuple(args.comparison_tile_shape), comparison_halo=tuple(args.comparison_halo),
    )


if __name__ == "__main__":
    main()