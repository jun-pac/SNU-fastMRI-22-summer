"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
import sys

import pytorch_lightning as pl
sys.path.insert(0,'/root')

from fastMRI.utils.data.mri_data import fetch_dir
from fastMRI.utils.pl_modules import VarNetModule
from fastMRI.utils.data.subsample import create_mask_for_mask_type
from pytorch_lightning import loggers as pl_loggers
from fastMRI.utils.data.DA_transforms import VarNetDataTransform
from fastMRI.utils.mraugment.data_augment import DataAugmentor
from fastMRI.utils.pl_modules import FastMriDataModule


def cli_main(args):
    pl.seed_everything(args.seed)
    
    # ------------
    # model
    # ------------
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        alpha = args.alpha
    )
    
    
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations, args.offset
    )
    
    # use random masks for train transform, fixed masks for val transform
    current_epoch_fn = lambda: model.current_epoch
    augmentor=DataAugmentor(args, current_epoch_fn)
    
    train_transform = VarNetDataTransform(augmentor=augmentor, mask_func=None, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=None)
    test_transform = VarNetDataTransform()
    
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        is_pruno=False
    )


    # ------------
    # trainer
    # ------------
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer.from_argparse_args(args,logger=tb_logger)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("/root/fastMRI/fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("brain_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "fastMRI" / "train_model"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[5],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--offset",
        nargs="+",
        default=3,
        type=int,
        help="offset to use for masks",
    )


    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="equispaced",  # VarNet uses equispaced mask
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # data augmentation config
    parser = DataAugmentor.add_augmentation_specific_args(parser)
    
    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=6, #6,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=13,  #13 # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=5,  #5 # number of top-level channels for sense est. U-Net
        lr=0.001,  # Adam learning rate
        lr_step_size=10,  # epoch at which to decrease learning rate
        lr_gamma=0.5,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=100,  # max number of epochs
        gradient_clip_val=1
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=100, #3,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]
    
    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()