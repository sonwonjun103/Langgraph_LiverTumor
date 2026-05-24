def build_sam_adapter(args):
    raise RuntimeError(
        "3DSAMAdapter is trained with train.sam_adapter_trainer.train_sam_adapter "
        "because it uses separate image encoder, prompt encoders, and mask decoder. "
        "Run: python main.py --model sam_adapter --roi_size 128 128 128 "
        "--sam_checkpoint ckpt/sam_vit_b_01ec64.pth "
        "--sam_pretrained_ckpt snapshot/lits/last.pth.tar"
    )
