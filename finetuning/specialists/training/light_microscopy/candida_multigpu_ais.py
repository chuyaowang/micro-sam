"""Multi-GPU (DDP) fine-tuning of the SAM encoder + AIS decoder on Candida albicans data.

This trains a UNETR (SAM image encoder backbone + automatic-instance-segmentation decoder)
without any interactive / prompt-based components, distributed across all local GPUs
(e.g. Kaggle's 2x T4) via ``torch_em.multi_gpu_training.train_multi_gpu``.

Design notes
------------
* **Pre-processing is decoupled.** This script consumes already-pre-processed images as
  ``.tif`` files (one image per file, channels-first, uint8 or uint16). All of the
  ``ImageContainer`` work (DIC-shift correction, rescaling, quantization, channel merging)
  is done *once* in the notebook/main process and written to ``--raw-dir`` / ``--label-dir``.
  This avoids re-running pre-processing in every spawned DDP child and sidesteps the
  ``mp.spawn`` re-import problem for notebook-defined classes.

* **Model setup reuses micro_sam internals.** ``build_unetr_model`` is a thin factory around
  the exact two functions that ``train_instance_segmentation`` uses
  (``get_trainable_sam_model`` + ``get_unetr``), so there is no duplicated model code.

* **Mixed precision.** ``train_multi_gpu`` uses fp16 autocast. This is safe on T4 (Turing
  TU104, has Tensor Cores -> fp32 accumulation). The bf16 work-around is only needed on
  Tensor-Core-less GPUs (e.g. GTX 16xx) and is intentionally not used here.

* **k-fold cross-validation.** ``train_multi_gpu`` is a single training session, so CV runs it
  once per fold in a sequential loop. Each fold tears down its process group before the next
  starts (the hard-coded ``MASTER_PORT`` is safe to reuse sequentially). Pass ``--fold`` to run
  a single fold (useful to fit inside Kaggle's session time limit and resume later).

Example (Kaggle, 2x T4, internet on, fork installed via ``pip install -e``)::

    !python candida_multigpu_ais.py \
        --raw-dir /kaggle/working/preprocessed/raw \
        --label-dir /kaggle/working/preprocessed/labels \
        --encoder /kaggle/input/.../vit_b_lm \
        --decoder /kaggle/input/.../vit_b_lm_decoder \
        --save-root /kaggle/working/model_checkpoints \
        --model-type vit_b_lm \
        --patch-shape 512 512 \
        --n-folds 6
"""

import os
import csv
import glob
import shutil
import argparse
import statistics
from typing import List, Optional, Tuple

import numpy as np
import torch
import kornia.augmentation as K

import torch_em
from torch_em.data.sampler import MinInstanceSampler
from torch_em.transform.augmentation import KorniaAugmentationPipeline
from torch_em.multi_gpu_training import train_multi_gpu
from torch_em.trainer.tensorboard_logger import TensorboardLogger
from torch_em.util import load_image

from micro_sam.training import default_sam_dataset
from micro_sam.training.util import get_trainable_sam_model, require_8bit
from micro_sam.instance_segmentation import get_unetr


# ---------------------------------------------------------------------------
# Per-rank TensorBoard logging + loss-spike image capture (module-level so it
# survives the mp.spawn re-import).
# ---------------------------------------------------------------------------
class RankTensorboardLogger(TensorboardLogger):
    """Per-rank TensorBoard logging plus on-demand image capture when the loss spikes.

    Two behaviors on top of torch_em's ``TensorboardLogger``:

    1. **Per-rank subfolders.** The base logger derives its log dir from ``trainer.name``
       only (no rank component), so every DDP rank's SummaryWriter targets the same
       directory and TensorBoard merges their event files into one noisy run. We
       temporarily suffix ``trainer.name`` while the parent derives ``log_dir``, then
       restore it so checkpoint paths (which also use ``name``) are unaffected. Single-GPU
       runs (``rank is None``) keep the base directory.

    2. **Loss-spike images.** The base logger only writes prediction images on the periodic
       ``log_image_interval``, so a loss spike *between* intervals is never imaged. When the
       train/val loss exceeds ``spike_image_threshold`` we additionally log input/target/
       prediction under a dedicated ``train_spike/`` or ``validation_spike/`` tag, so spikes
       are easy to find and inspect. (Off-interval frames carry no gradient overlay, since
       the trainer only ``retain_grad()``s the prediction at the periodic interval.)
    """

    def __init__(self, trainer, save_root, spike_image_threshold=None, **kwargs):
        self.spike_image_threshold = spike_image_threshold
        rank = getattr(trainer, "rank", None)
        if rank is None:  # single-GPU path: behave exactly like the base logger
            super().__init__(trainer, save_root, **kwargs)
            return
        original_name = trainer.name
        trainer.name = os.path.join(original_name, f"rank{rank}")
        try:
            super().__init__(trainer, save_root, **kwargs)
        finally:
            trainer.name = original_name

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        super().log_train(step, loss, lr, x, y, prediction, log_gradients)
        loss_value = loss.item() if hasattr(loss, "item") else float(loss)
        if self.spike_image_threshold is not None and loss_value > self.spike_image_threshold:
            self.log_images(step, x, y, prediction, "train_spike")

    def log_validation(self, step, metric, loss, x, y, prediction):
        super().log_validation(step, metric, loss, x, y, prediction)
        loss_value = loss.item() if hasattr(loss, "item") else float(loss)
        if self.spike_image_threshold is not None and loss_value > self.spike_image_threshold:
            self.log_images(step, x, y, prediction, "validation_spike")


# ---------------------------------------------------------------------------
# DDP-aware trainer: synchronize the validation metric across ranks
# (module-level so it survives the mp.spawn re-import / pickle).
# ---------------------------------------------------------------------------
class SyncedValTrainer(torch_em.trainer.DefaultTrainer):
    """``DefaultTrainer`` that all-reduces the validation metric to the mean over all ranks.

    ``torch_em.multi_gpu_training`` wraps the val dataset in a ``DistributedSampler``, so each
    rank validates on a *disjoint* shard of the tile grid (rank 0 -> tiles 0, 2, 4, ...; rank 1
    -> tiles 1, 3, 5, ...) and ``_validate_impl`` returns the mean over that rank's shard only.
    The early-stopping, checkpoint-selection and ``ReduceLROnPlateau`` decisions in ``fit`` are
    then made independently per rank from *different* metric values, so the ranks can reach the
    early-stopping threshold on different epochs. When one rank breaks out of the epoch loop while
    the other enters the next training step, that step's gradient all-reduce has no partner and
    dead-locks until the 600 s NCCL watchdog aborts the process (``OpType=ALLREDUCE`` timeout).

    Averaging the metric across ranks makes every rank compute the same value -- the mean over the
    *full* tile grid -- so all of those decisions stay in lock-step (no desync, no deadlock). It
    also fixes a quieter bug: the reported/saved ``best_metric`` was previously only rank 0's half
    of the tiles.
    """

    def _validate_impl(self, forward_context):
        metric = super()._validate_impl(forward_context)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            metric_tensor = torch.tensor([metric], dtype=torch.float32, device=self.device)
            torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.AVG)
            metric = metric_tensor.item()
        return metric


# ---------------------------------------------------------------------------
# Deterministic, cell-rich validation patch (module-level for mp.spawn).
# ---------------------------------------------------------------------------
def _best_variance_patch(raw: np.ndarray, patch_shape) -> Tuple[slice, slice]:
    """Return the spatial bounding box of the highest-variance non-overlapping tile.

    The image is divided into a grid of non-overlapping ``patch_shape`` windows and each
    tile is scored by its intensity variance. Bright cells over dark background produce a
    large spread (high-intensity cell pixels + low-intensity background pixels), so the
    max-variance tile is likely to contain cells -- unlike a blind center crop, which can
    land on empty background. Returns ``(slice_y, slice_x)`` over the spatial axes only.
    """
    pH, pW = int(patch_shape[-2]), int(patch_shape[-1])
    channel_first = raw.ndim == 3 and raw.shape[-1] > 16
    if raw.ndim == 3 and not channel_first:      # channel-last (H, W, C)
        H, W = raw.shape[0], raw.shape[1]
    elif raw.ndim == 3:                          # channel-first (C, H, W)
        H, W = raw.shape[1], raw.shape[2]
    else:                                        # (H, W)
        H, W = raw.shape

    ny, nx = max(1, H // pH), max(1, W // pW)
    best, best_score = (0, 0), -np.inf
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * pH, ix * pW
            if channel_first:
                tile = raw[:, y0:y0 + pH, x0:x0 + pW]
            elif raw.ndim == 3:
                tile = raw[y0:y0 + pH, x0:x0 + pW, :]
            else:
                tile = raw[y0:y0 + pH, x0:x0 + pW]
            score = float(np.var(tile))
            if score > best_score:
                best_score, best = score, (y0, x0)

    y0, x0 = best
    return slice(y0, y0 + pH), slice(x0, x0 + pW)


def fixed_crop_val_dataset(**kwargs):
    """Build a validation dataset that always uses ONE fixed, cell-rich patch per image.

    Drop-in replacement for ``default_sam_dataset`` as the validation dataset callable. For
    each image it loads the array, picks the highest-variance ``patch_shape`` tile
    (`_best_variance_patch`), and crops both raw and label to that window. Because the
    cropped array equals ``patch_shape``, the dataset's own random crop becomes a no-op
    (``shape - patch_shape == 0``), so validation sees the **same patch every time**. A
    val-loss spike then reflects model state, not which crop got drawn. The instance
    ``sampler`` is dropped (a fixed patch would loop forever against the rejection sampler).
    """
    raw_paths = kwargs.pop("raw_paths")
    label_paths = kwargs.pop("label_paths")
    kwargs.pop("raw_key", None)
    kwargs.pop("label_key", None)
    kwargs.pop("sampler", None)
    patch_shape = kwargs["patch_shape"]

    if not isinstance(raw_paths, (list, tuple)):
        raw_paths, label_paths = [raw_paths], [label_paths]

    cropped_raw, cropped_label = [], []
    for rp, lp in zip(raw_paths, label_paths):
        raw = np.asarray(load_image(rp))
        label = np.asarray(load_image(lp))
        sy, sx = _best_variance_patch(raw, patch_shape)
        if raw.ndim == 3 and raw.shape[-1] > 16:      # channel-first (C, H, W)
            cropped_raw.append(raw[:, sy, sx])
        elif raw.ndim == 3:                           # channel-last (H, W, C)
            cropped_raw.append(raw[sy, sx, :])
        else:
            cropped_raw.append(raw[sy, sx])
        cropped_label.append(label[sy, sx])

    return default_sam_dataset(
        raw_paths=cropped_raw, raw_key=None,
        label_paths=cropped_label, label_key=None,
        sampler=None, **kwargs,
    )


def _no_augmentation(raw, labels):
    """Identity transform: disable all augmentation on the validation tiles.

    ``default_sam_dataset`` substitutes the default (flip + 90-degree rotation) pipeline when
    ``transform is None``, so the deterministic validation tiles would still be randomly
    rotated/flipped unless an explicit no-op is supplied. Kept at module level (not a lambda)
    so it survives the ``mp.spawn`` pickle on the DDP worker path.
    """
    return raw, labels


def _grid_slices(raw: np.ndarray, patch_shape) -> List[Tuple[slice, slice]]:
    """Return spatial slices for every full, non-overlapping ``patch_shape`` tile of ``raw``.

    Edge remainders smaller than ``patch_shape`` are dropped so each tile is exactly the patch
    size (the dataset's own random crop then becomes a no-op). Channel layout is detected with
    the same ``shape[-1] > 16`` heuristic as ``_best_variance_patch``.
    """
    pH, pW = int(patch_shape[-2]), int(patch_shape[-1])
    if raw.ndim == 3 and raw.shape[-1] > 16:      # channel-first (C, H, W)
        H, W = raw.shape[1], raw.shape[2]
    elif raw.ndim == 3:                           # channel-last (H, W, C)
        H, W = raw.shape[0], raw.shape[1]
    else:                                         # (H, W)
        H, W = raw.shape
    slices = []
    for iy in range(max(1, H // pH)):
        for ix in range(max(1, W // pW)):
            y0, x0 = iy * pH, ix * pW
            slices.append((slice(y0, y0 + pH), slice(x0, x0 + pW)))
    return slices


def tiled_val_dataset(**kwargs):
    """Validation dataset that deterministically covers the WHOLE image with a tile grid.

    Drop-in replacement for ``default_sam_dataset`` as the validation dataset callable. Each
    image is split into all full, non-overlapping ``patch_shape`` tiles -- background-heavy,
    cell-heavy and balanced alike -- so the validation metric is representative of every region
    type, not just one cell-rich patch. With augmentation disabled (the caller passes
    ``transform=_no_augmentation``) and the loader's ``shuffle`` off under DDP, every tile is
    visited once per validation pass, so the metric is both representative and reproducible. The
    instance ``sampler`` is dropped so empty/background tiles are kept (they test the model's
    ability to predict empty masks).
    """
    raw_paths = kwargs.pop("raw_paths")
    label_paths = kwargs.pop("label_paths")
    kwargs.pop("raw_key", None)
    kwargs.pop("label_key", None)
    kwargs.pop("sampler", None)
    patch_shape = kwargs["patch_shape"]

    if not isinstance(raw_paths, (list, tuple)):
        raw_paths, label_paths = [raw_paths], [label_paths]

    raw_tiles, label_tiles = [], []
    for rp, lp in zip(raw_paths, label_paths):
        raw = np.asarray(load_image(rp))
        label = np.asarray(load_image(lp))
        channel_first = raw.ndim == 3 and raw.shape[-1] > 16
        for sy, sx in _grid_slices(raw, patch_shape):
            if channel_first:
                raw_tiles.append(raw[:, sy, sx])
            elif raw.ndim == 3:
                raw_tiles.append(raw[sy, sx, :])
            else:
                raw_tiles.append(raw[sy, sx])
            label_tiles.append(label[sy, sx])

    # One sample per tile so a validation pass covers the whole image exactly once.
    kwargs.setdefault("n_samples", len(raw_tiles))
    return default_sam_dataset(
        raw_paths=raw_tiles, raw_key=None,
        label_paths=label_tiles, label_key=None,
        sampler=None, **kwargs,
    )


# ---------------------------------------------------------------------------
# Augmentation (module-level so it survives the mp.spawn re-import / pickle).
# Mirrors the notebook's SplitPhotometricPipeline: geometric augs are applied to
# both raw and labels, photometric augs only to raw.
# ---------------------------------------------------------------------------
class SplitPhotometricPipeline(torch.nn.Module):
    """Apply geometric augmentations to (raw, labels) and photometric augmentations to raw only."""

    def __init__(self, geometric_augs, photometric_augs):
        super().__init__()
        self.geometric = KorniaAugmentationPipeline(*geometric_augs)
        self.photometric = KorniaAugmentationPipeline(*photometric_augs)

    def forward(self, raw, labels):
        raw, labels = self.geometric(raw, labels)
        # 90-degree rotation: same k for both raw and labels.
        k = torch.randint(0, 4, (1,)).item()
        raw = torch.rot90(raw, k, dims=[-2, -1])
        labels = torch.rot90(labels, k, dims=[-2, -1])
        raw, = self.photometric(raw)
        raw = raw.clamp(0, 255)  # bilinear interp in RandomAffine can exceed 255 by ~1e-5
        return raw, labels


def build_train_transform() -> SplitPhotometricPipeline:
    return SplitPhotometricPipeline(
        geometric_augs=[
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomAffine(degrees=0, scale=(0.85, 1.15), p=0.5),
        ],
        photometric_augs=[
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.5),
        ],
    )


# ---------------------------------------------------------------------------
# Model factory (module-level, built on CPU; train_multi_gpu moves it to the rank).
# ---------------------------------------------------------------------------
def build_unetr_model(
    model_type: str,
    checkpoint_path: Optional[str],
    decoder_path: Optional[str],
    freeze: Optional[List[str]],
    strict_decoder_loading: bool = True,
) -> torch.nn.Module:
    """Build the SAM-encoder + AIS-decoder UNETR using existing micro_sam functions.

    This is exactly the model that ``train_instance_segmentation`` builds internally,
    factored out so it can be passed to ``train_multi_gpu`` as the ``model_callable``.
    The model is built on CPU; ``train_multi_gpu._train_impl`` calls ``.to(rank)`` and
    wraps it in DistributedDataParallel.
    """
    sam_model, state = get_trainable_sam_model(
        model_type=model_type,
        device="cpu",
        checkpoint_path=checkpoint_path,
        return_state=True,
        freeze=freeze,
        decoder_path=decoder_path,  # forwarded to get_sam_model -> populates state["decoder_state"]
    )
    model = get_unetr(
        image_encoder=sam_model.sam.image_encoder,
        decoder_state=state.get("decoder_state", None),
        device="cpu",
        flexible_load_checkpoint=not strict_decoder_loading,
    )
    return model


# ---------------------------------------------------------------------------
# Data discovery and fold splitting.
# ---------------------------------------------------------------------------
def _discover_pairs(raw_dir: str, label_dir: str) -> Tuple[List[str], List[str]]:
    """Find matching raw/label .tif pairs, sorted by filename for deterministic ordering."""
    raw_paths = sorted(glob.glob(os.path.join(raw_dir, "*.tif")) + glob.glob(os.path.join(raw_dir, "*.tiff")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")) + glob.glob(os.path.join(label_dir, "*.tiff")))
    if len(raw_paths) == 0:
        raise FileNotFoundError(f"No .tif files found in raw dir: {raw_dir}")
    if len(raw_paths) != len(label_paths):
        raise ValueError(
            f"Mismatched raw/label counts: {len(raw_paths)} raw vs {len(label_paths)} labels. "
            "Ensure each raw image has exactly one corresponding label file and that sorting aligns them."
        )
    return raw_paths, label_paths


def make_fold(
    raw_paths: List[str], label_paths: List[str], n_folds: int, fold: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split into train/val for a given fold. n_folds == len(images) gives leave-one-out."""
    n = len(raw_paths)
    if not (1 <= n_folds <= n):
        raise ValueError(f"--n-folds must be in [1, {n}] (number of images), got {n_folds}.")
    if not (0 <= fold < n_folds):
        raise ValueError(f"--fold must be in [0, {n_folds - 1}], got {fold}.")

    # Contiguous index groups; fold `fold` is the validation group.
    val_idx = list(range(fold, n, n_folds)) if n_folds < n else [fold]
    val_set = set(val_idx)
    train_idx = [i for i in range(n) if i not in val_set]

    raw_train = [raw_paths[i] for i in train_idx]
    label_train = [label_paths[i] for i in train_idx]
    raw_val = [raw_paths[i] for i in val_idx]
    label_val = [label_paths[i] for i in val_idx]
    return raw_train, label_train, raw_val, label_val


# ---------------------------------------------------------------------------
# Held-out before/after comparison (runs in the main process after each fold,
# independent of the DDP workers). Loads the original model from the saved
# encoder/decoder weights and the fine-tuned model from the fold's best.pt,
# runs whole-image AIS on the held-out image, scores both with mean
# segmentation accuracy, and saves a 2x6 before/after figure.
# ---------------------------------------------------------------------------
def _to_channels_last(image: np.ndarray) -> np.ndarray:
    """Channels-first ``(C, H, W)`` -> ``(H, W, C)``; leave grayscale / channels-last as-is.

    ``micro_sam.util._to_image`` treats the *last* axis as channels, so the pre-processed
    channels-first raw tiff must be transposed before inference.
    """
    if image.ndim == 3 and image.shape[-1] > 16:  # (C, H, W): last axis is spatial -> channels first
        return np.ascontiguousarray(np.transpose(image, (1, 2, 0)))
    return np.ascontiguousarray(image)


def _safe_load_checkpoint(checkpoint_path):
    """``torch.load`` a trainer checkpoint without needing the symbols pickled into its ``init`` blob.

    ``DefaultTrainer.save_checkpoint`` stores an ``init`` entry that pickles the train/val ``Dataset``
    objects, and therefore their transforms (e.g. ``SplitPhotometricPipeline``, ``_no_augmentation``).
    Those transforms were defined in a training script that ran as ``__main__`` (``python
    candida_*.py``), so pickle recorded their class path as ``__main__.<Name>`` -- unresolvable when
    the checkpoint is loaded from a *different* process such as a notebook, raising
    ``AttributeError: Can't get attribute '<Name>' on <module '__main__'>``. Callers here only consume
    ``model_state`` / scalar fields, so any global that cannot be imported is replaced with a harmless
    placeholder. Those placeholders live only inside the ``init`` blob, which the checkpoint-stripping
    helpers drop before re-saving, so they are never used or persisted.
    """
    import pickle
    import types

    class _MissingGlobal:
        """Stand-in for an unresolvable pickled class/function; reconstructs without side effects."""
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            pass

    class _TolerantUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _MissingGlobal

    # torch inspects ``pickle_module.__name__`` (to special-case dill), so the shim needs one.
    shim = types.SimpleNamespace(Unpickler=_TolerantUnpickler, load=pickle.load, __name__="candida_tolerant_pickle")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False, pickle_module=shim)


def _strip_ddp_prefix(checkpoint_path, out_dir):
    """Return a *portable* checkpoint whose ``model_state`` keys have no ``module.`` (DDP) prefix.

    ``train_multi_gpu`` trains a ``DistributedDataParallel``-wrapped model, so ``best.pt``'s
    ``model_state`` keys are prefixed ``module.`` (e.g. ``module.encoder.pos_embed``).
    ``export_instance_segmentation_model`` filters for keys starting with ``encoder`` and would
    miss them. Always writes a cleaned copy into ``out_dir`` and returns its path: the ``module.``
    prefix is stripped from ``model_state`` (a no-op for single-GPU checkpoints) and the trainer
    ``init`` blob is dropped. Dropping ``init`` is what makes the copy portable -- it pickles the
    datasets and their transforms, often under ``__main__`` (see ``_safe_load_checkpoint``), so
    keeping it would re-introduce the cross-process unpickling error on any later load.
    """
    from collections import OrderedDict

    state = _safe_load_checkpoint(checkpoint_path)
    state.pop("init", None)  # drop the un-portable trainer-init blob (datasets + their transforms)

    model_state = state.get("model_state", None)
    if model_state is not None and all(k.startswith("module.") for k in model_state):
        state["model_state"] = OrderedDict((k[len("module."):], v) for k, v in model_state.items())

    normalized_path = os.path.join(out_dir, "_cv_best_no_ddp.pt")
    torch.save(state, normalized_path)
    return normalized_path


def _load_predictor_and_segmenter(model_type, checkpoint_path, decoder_path, device, is_tiled):
    """Build a SAM predictor + AIS segmenter from saved weights."""
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


def _run_ais(predictor, segmenter, image, tile_shape, halo) -> dict:
    """Run whole-image AIS and return the instance map plus the decoder's three output maps."""
    from micro_sam import util

    is_tiled = tile_shape is not None
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor, input_=image, ndim=2, tile_shape=tile_shape, halo=halo,
    )
    init_kwargs = dict(image=image, image_embeddings=image_embeddings)
    # output_mode="instance_segmentation" returns the label image directly.
    generate_kwargs = {"output_mode": "instance_segmentation"}
    if is_tiled:
        init_kwargs["batch_size"] = 1
        generate_kwargs.update(tile_shape=tile_shape, halo=halo)
    segmenter.initialize(**init_kwargs)
    instances = segmenter.generate(**generate_kwargs)
    state = segmenter.get_state()
    return {
        "instances": instances,
        "foreground": state["foreground"],
        "center_distances": state["center_distances"],
        "boundary_distances": state["boundary_distances"],
    }


def _save_comparison_figure(raw_image, gt_labels, results, msa, out_path, model_type, stem):
    """2x6 grid: rows = (Original, Fine-tuned); cols = Raw, GT, Instances, Foreground, Center, Boundary."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no interactive display in the training process
    import matplotlib.pyplot as plt
    from torch_em.util.util import get_random_colors

    disp_raw = raw_image if raw_image.ndim == 2 else raw_image[..., :3]
    col_titles = ["Raw", "Ground truth", "Instances", "Foreground prob", "Center distance", "Boundary distance"]
    fig, axes = plt.subplots(2, 6, figsize=(28, 9))
    for r, key in enumerate(("original", "finetuned")):
        res = results[key]
        inst = res["instances"]
        axes[r, 0].imshow(disp_raw, cmap="gray" if disp_raw.ndim == 2 else None)
        axes[r, 0].set_ylabel(f"{key.capitalize()}\nmSA={msa[key]:.3f}", fontsize=12)
        axes[r, 1].imshow(gt_labels, cmap=get_random_colors(gt_labels), interpolation="nearest")
        axes[r, 2].imshow(inst, cmap=get_random_colors(inst), interpolation="nearest")
        axes[r, 3].imshow(res["foreground"], cmap="viridis")
        axes[r, 4].imshow(res["center_distances"], cmap="magma")
        axes[r, 5].imshow(res["boundary_distances"], cmap="magma")
        axes[r, 2].set_title(f"Instances (n={int(inst.max())})")
        for c in range(6):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r == 0:
            for c in (0, 1, 3, 4, 5):
                axes[r, c].set_title(f"Ground truth (n={int(gt_labels.max())})" if c == 1 else col_titles[c])
    fig.suptitle(f"Held-out before/after - {stem} - {model_type}", fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compare_held_out(
    raw_path, label_path, model_type, encoder, decoder, best_checkpoint, figure_dir,
    device=None, tile_shape=(512, 512), halo=(64, 64),
):
    """Segment the held-out image with the original and fine-tuned model; save figure, return mSA.

    Exports the fold's ``best.pt`` to an AIS-ready model (DDP prefix stripped + image encoder
    remapped + ``decoder_state``), runs tiled whole-image AIS with both the original
    (encoder/decoder) and fine-tuned model, scores each against the ground truth with mean
    segmentation accuracy, and writes a 2x6 before/after figure. Returns
    ``{"original": msa, "finetuned": msa}`` (or None if the checkpoint is missing).
    """
    import micro_sam.training as sam_training
    from micro_sam.util import get_device
    from elf.evaluation import mean_segmentation_accuracy

    if not os.path.exists(best_checkpoint):
        print(f"[cv-comparison] skipped: no checkpoint at {best_checkpoint}")
        return None

    device = get_device(device)
    is_tiled = tile_shape is not None
    raw_image = _to_channels_last(np.asarray(load_image(raw_path)))
    gt_labels = np.asarray(load_image(label_path))
    os.makedirs(figure_dir, exist_ok=True)

    # Export the fine-tuned model (strip DDP prefix first so the encoder filter matches).
    trained_path = _strip_ddp_prefix(best_checkpoint, figure_dir)
    export_path = os.path.join(figure_dir, "_cv_export.pth")
    sam_training.export_instance_segmentation_model(
        trained_model_path=trained_path, output_path=export_path,
        model_type=model_type, initial_checkpoint_path=encoder,
    )
    if trained_path != best_checkpoint and os.path.exists(trained_path):
        os.remove(trained_path)

    results, msa = {}, {}
    runs = (
        ("original", encoder, decoder),     # saved pretrained weights (decoder is a separate file)
        ("finetuned", export_path, None),   # exported best.pt (decoder lives inside the checkpoint)
    )
    for key, checkpoint_path, decoder_path in runs:
        predictor, segmenter = _load_predictor_and_segmenter(
            model_type, checkpoint_path, decoder_path, device, is_tiled,
        )
        results[key] = _run_ais(predictor, segmenter, raw_image, tile_shape, halo)
        msa[key] = float(mean_segmentation_accuracy(results[key]["instances"], gt_labels))
        del predictor, segmenter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(export_path):
        os.remove(export_path)  # ~400 MB; the figure + mSA are what we keep

    stem = os.path.splitext(os.path.basename(str(raw_path)))[0]
    out_path = os.path.join(figure_dir, f"comparison_{stem}.png")
    _save_comparison_figure(raw_image, gt_labels, results, msa, out_path, model_type, stem)
    print(f"[cv-comparison] {stem}: mSA original={msa['original']:.4f} -> finetuned={msa['finetuned']:.4f}")
    print(f"[cv-comparison] saved figure to: {out_path}")
    return msa


# ---------------------------------------------------------------------------
# Persist per-fold results so the mSA / val-loss numbers survive the
# one-fold-per-subprocess workflow (each fold appends its own row).
# ---------------------------------------------------------------------------
def _write_cv_result(csv_path, fold, val_stems, result):
    """Append (or replace) this fold's results row in the CV results CSV.

    Columns: ``fold, val_images, msa_original, msa_finetuned, tiled_val_loss``. Each fold writes
    its own row as soon as it finishes, so the table survives the one-fold-per-subprocess workflow
    and partial runs (the printed summary, by contrast, only covers the folds run in *this*
    process). Re-running a fold replaces its existing row instead of duplicating it (matched on the
    fold index). Numeric fields are written with 6-decimal precision; missing values (e.g. mSA when
    ``--no-comparison`` is set, or the loss when no ``best.pt`` was produced) are left blank.
    """
    fieldnames = ["fold", "val_images", "msa_original", "msa_finetuned", "tiled_val_loss"]

    def _fmt(v):
        return "" if v is None else f"{v:.6f}"

    new_row = {
        "fold": str(fold),
        "val_images": ";".join(val_stems),
        "msa_original": _fmt(result["msa_original"]),
        "msa_finetuned": _fmt(result["msa_finetuned"]),
        "tiled_val_loss": _fmt(result["best_metric"]),
    }

    # Read existing rows (dropping any prior row for this fold), so a rerun replaces rather than
    # duplicates. Folds run sequentially (one subprocess at a time), so there is no write race.
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("fold") != str(fold)]
    rows.append(new_row)
    rows.sort(key=lambda r: int(r["fold"]))

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# Training driver.
# ---------------------------------------------------------------------------
def run_fold(args, fold: int, raw_paths: List[str], label_paths: List[str]) -> dict:
    """Run one DDP training session for a single fold. Returns the fold's best metric."""
    raw_train, label_train, raw_val, label_val = make_fold(raw_paths, label_paths, args.n_folds, fold)

    name = f"{args.name}_fold{fold}"
    print(f"\n{'=' * 70}\nFold {fold}/{args.n_folds - 1}: "
          f"{len(raw_train)} train / {len(raw_val)} val\n"
          f"  val image(s): {[os.path.basename(p) for p in raw_val]}\n{'=' * 70}")

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

    train_dataset_kwargs = dict(
        raw_paths=raw_train, label_paths=label_train,
        is_train=True, transform=build_train_transform(), **shared_ds_kwargs,
    )
    # Validation covers the whole held-out image with a deterministic tile grid (all region
    # types), augmentation disabled, so the metric is representative and reproducible.
    val_dataset_kwargs = dict(
        raw_paths=raw_val, label_paths=label_val,
        is_train=False, transform=_no_augmentation, **shared_ds_kwargs,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        # Under DDP (mp.spawn) workers re-import the whole stack (torch_em -> tensorflow/jax/wandb)
        # on creation. Keep them alive across epochs so that import cost is paid once, not per epoch.
        persistent_workers=args.num_workers > 0,
    )

    loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)

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
        val_dataset_callable=tiled_val_dataset,  # deterministic whole-image tile grid (see tiled_val_dataset)
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=int(args.iterations),
        # Every trainable parameter receives a gradient each step, so the per-iteration autograd
        # graph traversal that find_unused_parameters=True performs is pure overhead (and warns).
        # Only enable it when freezing, as a conservative guard for partially-used subgraphs.
        find_unused_parameters=args.freeze is not None,
        optimizer_callable=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=args.lr),
        lr_scheduler_callable=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs=dict(mode="min", factor=0.9, patience=3),
        # trainer params (forwarded to DefaultTrainer via **kwargs). SyncedValTrainer all-reduces
        # the per-rank validation metric so early stopping fires on the same epoch on every rank
        # (otherwise ranks desync and the surviving rank dead-locks in the gradient all-reduce).
        trainer_callable=SyncedValTrainer,
        logger=RankTensorboardLogger,  # each rank -> logs/<name>/rank<K>/ (separate TB runs)
        logger_kwargs=dict(spike_image_threshold=args.spike_image_threshold),
        name=name,
        save_root=args.save_root,
        loss=loss,
        metric=loss,
        early_stopping=args.early_stopping,
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False,
    )

    # Everything below runs in the main process after the DDP workers have exited.
    save_root = "" if args.save_root is None else args.save_root
    ckpt_dir = os.path.join(save_root, "checkpoints", name)
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    log_dir = os.path.join(save_root, "logs", name)

    # Read the fold's best (tiled-val) metric back before any checkpoint cleanup.
    best_metric = None
    if os.path.exists(best_ckpt):
        best_metric = _safe_load_checkpoint(best_ckpt).get("best_metric")
        print(f"Fold {fold}: best tiled-val metric = {best_metric:.6f}")
    else:
        print(f"Fold {fold}: no best.pt found at {best_ckpt}")

    # Before/after whole-image AIS on the held-out image(s): figure + mean segmentation accuracy.
    msa_original, msa_finetuned = [], []
    if not args.no_comparison:
        for rp, lp in zip(raw_val, label_val):
            try:
                m = compare_held_out(
                    raw_path=rp, label_path=lp, model_type=args.model_type,
                    encoder=args.encoder, decoder=args.decoder, best_checkpoint=best_ckpt,
                    figure_dir=log_dir,
                    tile_shape=tuple(args.comparison_tile_shape), halo=tuple(args.comparison_halo),
                )
            except Exception as e:
                import traceback
                print(f"[cv-comparison] skipped for {os.path.basename(rp)}: {e}")
                traceback.print_exc()
                m = None
            if m is not None:
                msa_original.append(m["original"])
                msa_finetuned.append(m["finetuned"])

    # Models are not needed -- keep only the logs and the comparison figures.
    if not args.keep_checkpoints and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"Fold {fold}: removed checkpoints ({ckpt_dir}); kept logs + figures.")

    result = {
        "best_metric": best_metric,
        "msa_original": statistics.mean(msa_original) if msa_original else None,
        "msa_finetuned": statistics.mean(msa_finetuned) if msa_finetuned else None,
    }

    # Persist this fold's row immediately, so the numbers survive the one-fold-per-subprocess
    # workflow even if a later fold crashes (the CSV is shared across folds via the base name).
    val_stems = [os.path.splitext(os.path.basename(p))[0] for p in raw_val]
    csv_path = os.path.join(save_root, f"{args.name}_cv_results.csv")
    _write_cv_result(csv_path, fold, val_stems, result)
    print(f"Fold {fold}: wrote CV results row -> {csv_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU (DDP) fine-tuning of SAM encoder + AIS decoder with k-fold CV."
    )
    parser.add_argument("--raw-dir", required=True, help="Directory of pre-processed raw .tif images (one per file).")
    parser.add_argument("--label-dir", required=True, help="Directory of label .tif images (one per file).")
    parser.add_argument("--encoder", default=None, help="Path to the SAM encoder checkpoint (vit_*_lm).")
    parser.add_argument("--decoder", default=None, help="Path to the AIS decoder checkpoint (vit_*_lm_decoder).")
    parser.add_argument("--save-root", default=None, help="Root dir for checkpoints/ and logs/.")
    parser.add_argument("--name", default="microsam_candida", help="Base run name; fold index is appended.")
    parser.add_argument("--model-type", default="vit_b_lm", help="SAM model type, e.g. vit_b_lm / vit_l_lm.")
    parser.add_argument("--patch-shape", type=int, nargs=2, default=[512, 512], help="Training patch shape H W.")
    parser.add_argument("--iterations", type=int, default=int(1e5), help="Max training iterations per fold.")
    parser.add_argument("--early-stopping", type=int, default=10, help="Early-stopping patience in epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per process.")
    parser.add_argument("--spike-image-threshold", type=float, default=1.0,
                        help="Also log input/target/prediction images to a *_spike/ tag whenever the "
                             "train/val loss exceeds this value (DiceBasedDistanceLoss ranges ~0-3).")
    parser.add_argument("--n-folds", type=int, default=None,
                        help="Number of CV folds. Default = number of images (leave-one-out).")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold index. If omitted, all folds run sequentially.")
    parser.add_argument("--freeze", type=str, nargs="+", default=None,
                        help="Model parts to freeze (e.g. image_encoder). Default: nothing frozen (encoder trained).")
    parser.add_argument("--flexible-decoder-loading", action="store_true",
                        help="Allow loading a decoder with mismatched output channels (reinitializes them).")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep each fold's checkpoints/ dir. Default: delete it after the comparison "
                             "figure is made, keeping only logs/ and the figures.")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip the per-fold held-out before/after AIS figure and mSA scoring.")
    parser.add_argument("--comparison-tile-shape", type=int, nargs=2, default=[512, 512],
                        help="Tile shape for the whole-image held-out AIS comparison inference.")
    parser.add_argument("--comparison-halo", type=int, nargs=2, default=[64, 64],
                        help="Per-tile overlap (halo) for the whole-image held-out AIS comparison inference.")
    args = parser.parse_args()

    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device visible. This script requires at least one GPU.")
    print(f"Visible GPUs: {torch.cuda.device_count()} "
          f"({', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))})")

    raw_paths, label_paths = _discover_pairs(args.raw_dir, args.label_dir)
    print(f"Discovered {len(raw_paths)} image/label pairs.")

    if args.n_folds is None:
        args.n_folds = len(raw_paths)  # leave-one-out by default

    folds = [args.fold] if args.fold is not None else list(range(args.n_folds))

    results = {}
    for fold in folds:
        results[fold] = run_fold(args, fold, raw_paths, label_paths)

    # Cross-validation summary: held-out mean segmentation accuracy (before -> after fine-tuning),
    # plus the tiled-val loss. mSA is the interpretable headline metric (higher is better).
    def _fmt(v, p=4):
        return "n/a" if v is None else f"{v:.{p}f}"

    finetuned_msa = [r["msa_finetuned"] for r in results.values() if r["msa_finetuned"] is not None]
    original_msa = [r["msa_original"] for r in results.values() if r["msa_original"] is not None]

    print(f"\n{'=' * 78}\nCross-validation summary ({len(folds)} folds)")
    print(f"  {'fold':>4}  {'mSA original':>13}  {'mSA finetuned':>13}  {'tiled-val loss':>14}")
    for fold in folds:
        r = results[fold]
        print(f"  {fold:>4}  {_fmt(r['msa_original']):>13}  {_fmt(r['msa_finetuned']):>13}  "
              f"{_fmt(r['best_metric'], 6):>14}")
    if finetuned_msa:
        o_mean = statistics.mean(original_msa) if original_msa else float("nan")
        f_mean = statistics.mean(finetuned_msa)
        f_std = statistics.stdev(finetuned_msa) if len(finetuned_msa) > 1 else 0.0
        print(f"\n  held-out mSA (higher is better): "
              f"original {o_mean:.4f} -> finetuned {f_mean:.4f} +/- {f_std:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()