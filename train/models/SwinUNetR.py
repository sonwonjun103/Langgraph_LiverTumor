import inspect

try:
    from monai.networks.nets import SwinUNETR
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "MONAI is required for SwinUNETR. Install it with: pip install monai"
    ) from exc


def build_swinunetr(args):
    # MONAI 1.5 removed img_size from SwinUNETR, while older versions still accept it.
    # Keep the builder version-tolerant so this project works across MONAI releases.
    kwargs = {
        "in_channels": 3,
        "out_channels": 1,
        "img_size": tuple(args.img_size),
        "feature_size": args.feature_size,
        "use_checkpoint": args.use_checkpoint,
        "spatial_dims": 3,
    }

    supported_args = inspect.signature(SwinUNETR).parameters
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_args
    }

    return SwinUNETR(**kwargs)
