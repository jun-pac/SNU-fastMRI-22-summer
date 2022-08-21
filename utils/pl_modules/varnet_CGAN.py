"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from .discriminator_CGAN import D_WGAN
from fastMRI.utils.data import transforms
from fastMRI.utils.models import VarNet
from .mri_module import MriModule


class VarNet_CGAN_Module(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055???3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 6,
        sens_pools: int = 4,
        sens_chans: int = 2,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        alpha: float = 128.0,
        lr_D: float = 0.01,
        adv_ratio: float = 0.05,
        iMSE_ratio: float = 100000000,
        fMSE_ratio: float = 10000000,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lr_D = lr_D
        self.adv_ratio = adv_ratio
        self.iMSE_ratio = iMSE_ratio
        self.fMSE_ratio = fMSE_ratio
        
        self.generator = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )
        self.gen_cnt=0
        self.lastD=0.0
        #pl_state_dict = torch.load('/root/varnet/pruno-nc=6/checkpoints/epoch=8.ckpt')['state_dict']
        
        pl_state_dict = torch.load('/root/varnet/SOTA/epoch=20-step=135870.ckpt')['state_dict']
        param_list = list(pl_state_dict.keys())
        state_dict = {}
        for name in param_list:
            if name[:6] == 'varnet':
                state_dict[name[7:]] = pl_state_dict[name]
        self.generator.load_state_dict(state_dict)
        
        
        self.discriminator = D_WGAN()
        self.loss = fastmri.SSIMLoss()
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCELoss()
        
    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.generator(masked_kspace, mask, num_low_frequencies)
    '''
    def generator_step(self,X,Y):
        d_output=self.discriminator(X,Y)
        Gloss = nn.BCELoss()(d_output, torch.ones(X.shape[0]))
        return Gloss
        
    def discriminator_step(self,X,Y):
        d_output=self.discriminator(X,Y)
        Dloss = nn.BCELoss()(d_output, torch.zeros(X.shape[0]))
        return Dloss
    '''
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        #print("batch_idx :",batch_idx)
        clip_value=0.01
        
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
        input = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace)), dim=1)
        
        # input = batch.grappa
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        input, output = transforms.center_crop_to_smallest(input, output)
        output = output * batch.image_mask
        
        
        if (optimizer_idx==0):
            d_output=self.discriminator(input.unsqueeze(0),output.unsqueeze(0))
            SSIMloss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
            iMSE=self.mseloss(output,target)
            outputc=torch.stack((output,torch.zeros(output.shape).to(torch.device('cuda:0'))),axis=-1)
            targetc=torch.stack((target,torch.zeros(target.shape).to(torch.device('cuda:0'))),axis=-1)
            fMSE=self.mseloss(fastmri.fft2c(outputc), fastmri.fft2c(targetc))
            #ADVloss=self.bceloss(d_output,torch.ones(input.shape[0]).to(torch.device('cuda:0')))
            ADVloss=d_output
            Gloss = -self.adv_ratio*ADVloss+self.iMSE_ratio*iMSE + self.fMSE_ratio*fMSE+SSIMloss
            
            self.log("train_SSIMloss", SSIMloss)
            self.log("train_iMSE", iMSE)
            self.log("train_fMSE", fMSE)
            self.log("train_advloss", ADVloss)
            self.log("train_Gloss", Gloss)
            
            if(self.gen_cnt%10==0):
                print(batch_idx,"(",self.gen_cnt,") th -","SSIM :",round(SSIMloss.item(),4),"|","iMSE :",round(iMSE.item()*self.iMSE_ratio,4),"|","fMSE :",round(fMSE.item()*self.fMSE_ratio,4),"|","ADV :", round(ADVloss.item()*self.adv_ratio,4), "|","G :", round(Gloss.item(),4),"|","D :", self.lastD)
            self.gen_cnt+=1
            return Gloss
        
        else :
            Dfake=self.discriminator(input.unsqueeze(0),output.unsqueeze(0))
            #loss_fake=self.bceloss(Dfake,torch.zeros(input.shape[0]).to(torch.device('cuda:0')))
            Dreal=self.discriminator(input.unsqueeze(0),target.unsqueeze(0))
            #loss_real=self.bceloss(Dreal,torch.ones(input.shape[0]).to(torch.device('cuda:0')))
            #Dloss=loss_fake+loss_real
            Dloss=Dfake-Dreal
            self.log("train_Dloss", Dloss)
            self.lastD=Dloss.item()
            return Dloss


    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        output = output * batch.image_mask

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            ),
            "batch_idx": 10
        }

    def test_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }
        
    def configure_optimizers(self):
        n_critic = 50
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=(0.5,0.999))
        '''g_scheduler = torch.optim.lr_scheduler.StepLR(
            g_optimizer, self.lr_step_size, self.lr_gamma
        )'''
        return (
            {'optimizer': g_optimizer, 'frequency': 1},
            {'optimizer': d_optimizer, 'frequency': n_critic}
        )
    

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--alpha",
            default=128.0,
            type=float,
            help="loss scaler",
        )

        return parser
