import os
"""Lightweight frame-level music injection into diffusion video latents."""
import math

import torch


class AudioSpatialInjector(torch.nn.Module):
    def __init__(self,latent_size):
        super().__init__()
        yy,xx=torch.meshgrid(torch.linspace(-1,1,latent_size),
                             torch.linspace(-1,1,latent_size),indexing='ij')
        radius=torch.sqrt(xx.square()+yy.square())
        basis=[torch.ones_like(xx),xx,yy]
        for center in (0.25,0.5,0.75,1.0):
            basis.append(torch.exp(-0.5*((radius-center)/0.13).square()))
        self.register_buffer('basis',torch.stack(basis),persistent=False)
        self.channels=len(basis)
        self.net=torch.nn.Sequential(
            torch.nn.Conv1d(22,128,kernel_size=3,padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(128,128,kernel_size=3,padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(128,4*self.channels,kernel_size=1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self,rhythm,melody,dynamics):
        x=torch.cat([rhythm,melody,dynamics],dim=-1).T[None]
        coefficients=self.net(x)[0].T.reshape(-1,4,self.channels)
        coefficients=0.6*torch.tanh(coefficients)
        delta=torch.einsum('fck,khw->cfhw',coefficients,self.basis)
        return delta[None]
