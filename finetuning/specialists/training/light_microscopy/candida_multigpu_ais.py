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
import glob
import argparse
import statistics
from typing import List, Optional, Tuple

import torch
import kornia.augmentation as K

import torch_em
from torch_em.data.sampler import MinInstanceSampler
from torch_em.transform.augmentation import KorniaAugmentationPipeline
from torch_em.multi_gpu_training import train_multi_gpu

from micro_sam.training import default_sam_dataset
from micro_sam.training.util import get_trainable_sam_model, require_8bit
from micro_sam.instance_segmentation import get_unetr


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
# Training driver.
# ---------------------------------------------------------------------------
def run_fold(args, fold: int, raw_paths: List[str], label_paths: List[str]) -> Optional[float]:
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
    val_dataset_kwargs = dict(
        raw_paths=raw_val, label_paths=label_val,
        is_train=False, **shared_ds_kwargs,
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
        val_dataset_callable=default_sam_dataset,
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=int(args.iterations),
        # With the encoder unfrozen every parameter receives a gradient, but keep this True so
        # the script also works when --freeze image_encoder is passed (DDP needs it then).
        find_unused_parameters=True,
        optimizer_callable=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=args.lr),
        lr_scheduler_callable=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs=dict(mode="min", factor=0.9, patience=3),
        # trainer params (forwarded to DefaultTrainer via **kwargs)
        trainer_callable=torch_em.trainer.DefaultTrainer,
        name=name,
        save_root=args.save_root,
        loss=loss,
        metric=loss,
        early_stopping=args.early_stopping,
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False,
    )

    # Read back the fold's best validation metric from the saved checkpoint.
    best_ckpt = os.path.join(
        "" if args.save_root is None else args.save_root, "checkpoints", name, "best.pt",
    )
    if os.path.exists(best_ckpt):
        best_metric = torch.load(best_ckpt, map_location="cpu", weights_only=False).get("best_metric")
        print(f"Fold {fold}: best metric = {best_metric:.6f}")
        return best_metric
    print(f"Fold {fold}: no best.pt found at {best_ckpt}")
    return None


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
    parser.add_argument("--n-folds", type=int, default=None,
                        help="Number of CV folds. Default = number of images (leave-one-out).")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold index. If omitted, all folds run sequentially.")
    parser.add_argument("--freeze", type=str, nargs="+", default=None,
                        help="Model parts to freeze (e.g. image_encoder). Default: nothing frozen (encoder trained).")
    parser.add_argument("--flexible-decoder-loading", action="store_true",
                        help="Allow loading a decoder with mismatched output channels (reinitializes them).")
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

    # Cross-validation summary.
    valid = [m for m in results.values() if m is not None]
    print(f"\n{'=' * 70}\nCross-validation summary ({len(valid)}/{len(folds)} folds completed)")
    for fold in folds:
        m = results[fold]
        print(f"  fold {fold}: {'n/a' if m is None else f'{m:.6f}'}")
    if len(valid) >= 1:
        mean = statistics.mean(valid)
        std = statistics.stdev(valid) if len(valid) > 1 else 0.0
        print(f"  mean +/- std: {mean:.6f} +/- {std:.6f}  (lower is better)")
    print("=" * 70)


if __name__ == "__main__":
    main()