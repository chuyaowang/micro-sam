"""Single-image overfit sanity check for SAM encoder + AIS decoder fine-tuning.

This is a debugging tool, not a training script. It trains the UNETR (SAM encoder + AIS
decoder) on a *single* image with augmentation disabled and validation == train, then checks
whether the loss collapses toward ~0. If the model cannot memorize one image, the bug is in the
data / model / loss wiring rather than in generalization, and there is no point debugging the
full training run until this passes.

It deliberately reuses ``torch_em.trainer.DefaultTrainer`` -- the exact trainer the real
multi-GPU run uses (via ``train_multi_gpu``) -- on a single GPU with no DDP. So this probe
exercises the real per-rank training step (same forward, loss application and logging) and the
TensorBoard output has the identical schema to the cross-validation runs
(``train/loss``, ``validation/loss``, ``validation/metric`` plus raw/target/prediction image
grids). The only layer it isolates out is DDP gradient synchronisation, which is intentional.

Runs on a single GPU, so it can be called directly from a notebook cell::

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

import torch

import torch_em
from torch_em.data.sampler import MinInstanceSampler

from micro_sam.util import get_device
from micro_sam.training import default_sam_loader
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


def _single_image_loader(raw_path, label_path, patch_shape, n_samples, is_train):
    """Single-image AIS loader with augmentation disabled."""
    return default_sam_loader(
        raw_paths=[str(raw_path)],
        label_paths=[str(label_path)],
        raw_key=None,
        label_key=None,
        patch_shape=tuple(patch_shape),
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        with_channels=True,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        is_train=is_train,
        raw_transform=require_8bit,
        transform=None,  # no augmentation: we want to memorize this exact image
        sampler=MinInstanceSampler(2, min_size=25),
        n_samples=n_samples,
    )


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
        device: Torch device. Defaults to the best available.

    Returns:
        Dict with best_metric, latest_metric, passed, name and log_dir.
    """
    device = get_device(device)

    if name is None:
        stem = os.path.splitext(os.path.basename(str(raw_path)))[0]
        name = f"overfit_{stem}"
    log_dir = os.path.join(save_root or ".", "logs", name)

    print(f"Overfit sanity check on:\n  raw:   {raw_path}\n  label: {label_path}\n"
          f"  device: {device}\n  tensorboard log_dir: {log_dir}")

    # Train and validation both point at the single image (validation == train).
    train_loader = _single_image_loader(raw_path, label_path, patch_shape, n_samples=iters_per_epoch, is_train=True)
    val_loader = _single_image_loader(raw_path, label_path, patch_shape, n_samples=max(1, iters_per_epoch // 5), is_train=False)

    model = _build_model(model_type, encoder, decoder, device)
    loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)

    trainer = torch_em.trainer.DefaultTrainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr),
        device=device,
        # fp32 so mixed precision is ruled out as a confound for this correctness probe.
        mixed_precision=False,
        log_image_interval=log_image_interval,
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
    args = parser.parse_args()

    overfit_single_image(
        raw_path=args.raw, label_path=args.label,
        encoder=args.encoder, decoder=args.decoder,
        model_type=args.model_type, patch_shape=tuple(args.patch_shape),
        n_iterations=args.iterations, lr=args.lr, pass_threshold=args.pass_threshold,
        iters_per_epoch=args.iters_per_epoch, log_image_interval=args.log_image_interval,
        save_root=args.save_root, name=args.name,
    )


if __name__ == "__main__":
    main()