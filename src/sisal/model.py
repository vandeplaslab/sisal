"""Model."""

import numpy as np
import torch.nn as nn


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


## Compute the output size for a given l_in size, kernel, stride and padding
def l_out(l_in, ker, stride, pad):
    return int(np.floor((l_in + 2 * pad - ker) / stride + 1))


class BetaVAE(nn.Module):
    def __init__(self, z_dim: int, in_size: int):
        super().__init__()
        if in_size < 19:
            raise ValueError("Input size too small for BetaVAE, should be at least 19")

        self.latent_std_min = 10  # so that the min variance is >=  e^{-10}
        self.z_dim = z_dim

        k1 = 10 if in_size % 2 == 0 else 9
        l1 = l_out(in_size, k1, 2, 1)

        k2 = 10 if l1 % 2 == 0 else 9

        l2 = l_out(l1, k2, 2, 1)

        # Param : q_\phi = N(\mu_\phi,diag(\sigma_\phi^2)) model diagonal covariance as diagonal
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 10, k1, stride=2, padding=1),  # 10 * 103
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 10, k2, stride=2, padding=1),  # 10 * 49
            nn.BatchNorm1d(10),
            nn.ReLU(),
            View((-1, 10 * l2)),  # 490
            nn.Linear(10 * l2, 200),  # 200
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, z_dim * 2),  # 10
        )

        # #p_\theta(x|z) = N(\mu, I)
        # input size = (1,z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 200),  # 200
            nn.ReLU(),
            nn.Linear(200, 10 * l2),  # 490
            nn.ReLU(),
            View((-1, 10, l2)),  # 10 * 49
            nn.ConvTranspose1d(10, 10, k2, stride=2, padding=1, output_padding=0),  # 10 * 103
            nn.ReLU(),
            nn.ConvTranspose1d(10, 1, k1, stride=2, padding=1, output_padding=0),  # 1 * 212
        )
        self.initialization()

    def forward(self, x):
        out_enc = self.encoder(x)
        z_mean = out_enc[:, : self.z_dim]
        z_logvar = out_enc[:, self.z_dim :]
        return (z_mean, z_logvar)

    def initialization(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    nn.init.zeros_(m.bias.data)
                elif isinstance(m, nn.Conv1d):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif isinstance(m, nn.ConvTranspose1d):
                    nn.init.xavier_normal_(m.weight.data)

    ##xavier_initialization
    def print_weight(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                    pass


class BetaVAESynthetic(BetaVAE):
    def __init__(self, z_dim: int, in_size: int = 5):
        super().__init__(z_dim, in_size)
        # Param : q_\phi = N(0,diag(\sigma^2)) model covariance as diagonal
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 10, 3, stride=1, padding=1),  # 10 * 5
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 10, 3, stride=1, padding=1),  # 10 * 5
            nn.BatchNorm1d(10),
            nn.ReLU(),
            View((-1, 10 * 5)),  # 490
            nn.Linear(10 * 5, 20),  # 200
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, z_dim * 2),  # 10
        )
        # p_\theta(x|z) = N(\mu, I)
        # input size = (1,z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 20),  # 200
            nn.ReLU(),
            nn.Linear(20, 10 * 5),  # 490
            nn.ReLU(),
            View((-1, 10, 5)),  # 10 * 49
            nn.ConvTranspose1d(10, 10, 3, stride=1, padding=1),  # 10 * 103
            nn.ReLU(),
            nn.ConvTranspose1d(10, 1, 3, stride=1, padding=1),  # 1 * 212
        )


# For compatibility with previous code
beta_vae = BetaVAE
beta_vae_synthetic = BetaVAESynthetic
