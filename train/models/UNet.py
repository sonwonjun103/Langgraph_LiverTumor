try:
    from monai.networks.nets import UNet
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "MONAI is required for UNet. Install it with: pip install monai"
    ) from exc


def build_unet(args):
    return UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        channels=tuple(args.channels),
        strides=tuple(args.strides),
        num_res_units=args.num_res_units,
        norm='batch',
        dropout=0.1
    )
