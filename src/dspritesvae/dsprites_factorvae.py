import torch
from torch import nn, distributions

from src.utils.model import Model
from src.dspritesvae.dsprites_vae import DspritesVAE


class DspritesFactorVAE(Model):
    def __init__(self):
        super(DspritesFactorVAE, self).__init__()
        self.z_dim = 10
        self.VAE = DspritesVAE()

        self.Discriminator = nn.Sequential(
            nn.Linear(self.z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 2),
        )

        self.xavier_initialization()

        # location to save model
        self.update_filepath()
    
    def __repr__(self):
        """
        String representation of class
        :return: string
        """
        return 'DspritesFactorVAE' + self.trainer_config

    def forward(self, x):
        return self.VAE(x)

    def encode(self, x):
        z_dist = self.VAE.encode(x)
        z_tilde, _, _ = self.VAE.reparametrize(z_dist)
        return z_tilde
    
    def forward_D(self, z):
        return self.Discriminator(z).squeeze()