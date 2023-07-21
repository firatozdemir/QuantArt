import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, StyleTransferModule
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

def disable_grad(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_model(model, pretrained_dict, key):
    model_dict = model.state_dict()
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith(key):
            new_dict[k[len(key)+1:]] = v
    model.load_state_dict(new_dict)

class VQModel_Ref(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key1="image1",
                 image_key2="image2",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=True,  # tell vector quantizer to return indices as bhw
                 checkpoint_encoder=None,
                 checkpoint_decoder=None,
                 use_residual=True,
                 use_conv=True,
                 use_selfatt=True,
                 ):
        super().__init__()
        self.image_key1 = image_key1
        self.image_key2 = image_key2
        self.encoder = Encoder(**ddconfig)
        self.encoder_real = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize_enc = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quantize_dec = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.quant_conv_real = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.model_x2y = StyleTransferModule(embed_dim, block_num=6, residual=use_residual, use_conv=use_conv, use_selfatt=use_selfatt)
    
        self.loss = instantiate_from_config(lossconfig)
        
        if checkpoint_encoder is not None and ckpt_path is None and os.path.isfile(checkpoint_encoder):
            print('loaded encoder chekpoint from', checkpoint_encoder)
            ckpt_enc = torch.load(checkpoint_encoder, map_location='cpu')['state_dict']
            load_model(self.encoder, ckpt_enc, 'encoder')
            load_model(self.quant_conv, ckpt_enc, 'quant_conv')
            try:
                load_model(self.quantize_enc, ckpt_enc, 'quantize')
            except:
                pass
        if checkpoint_decoder is not None and ckpt_path is None and os.path.isfile(checkpoint_decoder):
            print('loaded decoder chekpoint from', checkpoint_decoder)
            ckpt_dec = torch.load(checkpoint_decoder, map_location='cpu')['state_dict']
            load_model(self.encoder_real, ckpt_dec, 'encoder')
            load_model(self.decoder, ckpt_dec, 'decoder')
            load_model(self.quant_conv_real, ckpt_dec, 'quant_conv')
            load_model(self.post_quant_conv, ckpt_dec, 'post_quant_conv')
            try:
                load_model(self.quantize_dec, ckpt_dec, 'quantize')
            except:
                pass
        
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
      
    def encode(self, x, quantize=True):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if not quantize:
            return h, 0, [0,0,0]
        quant, emb_loss, info = self.quantize_enc(h)
        return quant, emb_loss, info

    def encode_real(self, x, quantize=True):
        h = self.encoder_real(x)
        h = self.quant_conv_real(h)
        if not quantize:
            return h, 0, [0,0,0]
        quant, emb_loss, info = self.quantize_dec(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def encode_to_z(self, x):
        quant_z, _, info = self.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    def transfer(self, x, ref, quantize=True):
        with torch.no_grad():  
            quant_x, _, _ = self.encode(x, quantize=True) 
            quant_x = quant_x.detach()
            quant_ref, _, info_ref = self.encode_real(ref, quantize=True)
            indices_ref = info_ref[2]
            quant_ref = quant_ref.detach()
        
        h_x = self.model_x2y(quant_x, quant_ref)
        if not quantize:
            return quant_x, h_x, quant_ref, torch.zeros(1).to(self.device), [0,0,0], indices_ref
        quant_y, diff_x2y, info_y = self.quantize_dec(h_x)
        indices_y = info_y[2]
        return quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x1 = self.get_input(batch, self.image_key1)
        x2 = self.get_input(batch, self.image_key2)
        
        quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref = self.transfer(x1, x2)

        if optimizer_idx == 0:
            # autoencode
            total_loss, aeloss, log_dict_ae = self.loss(diff_x2y, 
                                            quant_ref, quant_y,
                                            quant_x, 
                                            indices_ref, indices_y,
                                            optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return total_loss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(diff_x2y, 
                                            quant_ref, quant_y,
                                            quant_x, 
                                            indices_ref, indices_y,
                                            optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x1 = self.get_input(batch, self.image_key1)
        x2 = self.get_input(batch, self.image_key2)
        
        quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref = self.transfer(x1, x2)
        
        total_loss, aeloss, log_dict_ae = self.loss(diff_x2y, 
                                            quant_ref, quant_y, 
                                            quant_x, 
                                            indices_ref, indices_y,
                                            0, self.global_step, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(diff_x2y, 
                                            quant_ref, quant_y, 
                                            quant_x,
                                            indices_ref, indices_y,
                                            1, self.global_step, last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model_x2y.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.quantize_dec.embedding.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x1 = self.get_input(batch, self.image_key1)
        x2 = self.get_input(batch, self.image_key2)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref = self.transfer(x1, x2) # EDITED
        x2_out = self.decode(quant_y)
        log["vis"] = torch.cat((x1, x2, x2_out))
        return log


