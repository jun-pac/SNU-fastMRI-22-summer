'''
import argparse
from pathlib import Path
from utils.learning.test_part import forward

def parse():
    parser = argparse.ArgumentParser(description='Test Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-p', '--data_path', type=Path, default='../Data/image_Leaderboard/', help='Directory of test data')
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument("--input_key", type=str, default='image_input', help='Name of input key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.forward_dir = '../result' / args.net_name / 'reconstructions_forward' 
    print(args.forward_dir)
    forward(args)
'''
"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path
import sys
import os

import numpy as np
import requests
import torch
import tqdm

sys.path.insert(0,'/root')
from fastMRI.utils.data.mri_data import fetch_dir
from fastMRI.utils.data.transforms import VarNetDataTransform
from fastMRI.utils.pl_modules import FastMriDataModule, VarNetModule
from fastMRI.utils.data.subsample import create_mask_for_mask_type
from fastMRI.utils.models import VarNet
import fastMRI.utils.data.transforms as T
from fastMRI.utils.data import SliceDataset
import fastMRI.utils as fastMRI

VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def run_varnet_model(batch, model, device):
    crop_size = batch.crop_size

    output = model(batch.masked_kspace.to(device), batch.mask.to(device)).cpu()

    # detect FLAIR 203
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    output = T.center_crop(output, crop_size)[0]

    return output, int(batch.slice_num[0]), batch.fname[0]


def run_inference(challenge, state_dict_file, data_path, output_path, device):
    
    model = VarNet(num_cascades=6, pools=4, chans=13, sens_pools=4, sens_chans=5)
    pl_state_dict = torch.load(state_dict_file)['state_dict']
    param_list = list(pl_state_dict.keys())
    state_dict = {}
    for name in param_list:
        if name[:6] == 'varnet':
            state_dict[name[7:]] = pl_state_dict[name]
    
    model.load_state_dict(state_dict)
    model = model.eval()

    # data loader setup
    data_transform = T.VarNetDataTransform()
    dataset = SliceDataset(
        root=data_path, transform=data_transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm.tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_varnet_model(batch, model, device)
            
        outputs[fname].append((slice_num, output))

    # save outputs
    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    fastMRI.save_reconstructions(outputs, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="varnet_brain_mc",
        choices=(
            "varnet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default="/root/fastMRI/model/epoch=34.ckpt",
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        default="/root/input/leaderboard/kspace",
        type=Path,
        #required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="/root/output",
        #required=True,
        help="Path for saving reconstructions",
    )

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
    )