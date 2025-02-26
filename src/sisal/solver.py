import sys

import numpy as np

sys.path.insert(0, "/".join(sys.path[0].split("/")[0:-2]) + ("/src"))
# import model

import torch
import torch.nn.functional as F
from torch.optim import Adam

from sisal.model import BetaVAE
from sisal.utils import (
    compute_estimate_std,
    metric_disentangling,
    reparametrize,
)

# EPOCHS = 5
# best_vloss = 1_000_000.

# def reparametrize(mu, logvar):
#         std = logvar.div(2).exp()
#         eps = Variable(std.data.new(std.size()).normal_())
#         return mu + std*eps


class Solver:
    def __init__(self, args, in_size):
        # timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        # self.writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

        self.device = torch.device(args.device)
        self.z_dim = args.z_dim
        self.model = BetaVAE(args.z_dim, in_size).to(self.device)
        self.save_epochs = args.save_model_epochs
        self.EPOCHS = args.epochs
        self.train_bool = args.train
        self.PATH = "model/model_weights.pth"  ## Path to save the model
        self.beta = args.beta
        self.save_loss = args.save_loss

    ## Average KL over the batch
    def KL(self, mu, logvar):
        logvar = logvar.view(logvar.size(0), logvar.size(1))
        # KL = 0.5*(mu.pow(2).sum() - logvar.sum() + logvar.exp().sum() - d*batch_size).div(batch_size)
        klds = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        return klds.sum(1).mean(0)

    ##Average recons over the batch
    def reconstruction_loss(self, x, mu_x):
        # - \E_{z \sim q_\phi(z|x)}(\log p_\theta(x|z))
        # Model to start
        # Param : q_\phi = N(0,diag(\sigma^2)), p_\theta(z) = N(0,I) , p_\theta(x|z) = N(\mu, I)
        batch_size = x.size(0)

        r_loss = 0.5 * F.mse_loss(x, mu_x, reduction="sum").div(batch_size)
        return r_loss

    def loss(self, beta, x, z_mean, z_logvar, decoder_mean):
        return self.reconstruction_loss(x, decoder_mean) + beta * self.KL(z_mean, z_logvar)

    def train_one_epoch(self, epoch_index, dataloader):
        c = 0
        optim = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        running_loss = 0.0
        last_loss = 0.0

        running_recons = 0
        running_KL = 0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(self.device, non_blocking=True)
            z_mean, z_logvar = self.model.forward(x)
            z = reparametrize(z_mean, z_logvar)

            decoder_mean = self.model.decoder(z)
            beta_vae_loss = self.loss(self.beta, x, z_mean, z_logvar, decoder_mean)

            optim.zero_grad()
            beta_vae_loss.backward()
            optim.step()

            running_loss += beta_vae_loss.item()

            ##############
            reconstruction_loss = self.reconstruction_loss(x, decoder_mean)
            running_recons += reconstruction_loss
            kl_loss = self.KL(z_mean, z_logvar)
            running_KL += kl_loss
            ##############

            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                # tb_x = epoch_index * len(dataloader) + i + 1
                # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                # tb_writer.flush()
                running_loss = 0.0

                ###############
                last_recons = running_recons / 1000
                last_kl = running_KL / 1000
                running_recons = 0
                running_KL = 0
                ###############
            if i % 2000 == 1999 and self.save_epochs:
                torch.save(self.model, f"model/model_weight_n_{epoch_index * 3 + c}.pth")
                c += 1

        return last_loss, last_recons, last_kl

    def evaluate_loss(self, loader):
        with torch.no_grad():
            running_loss = 0.0
            running_recons = 0
            running_KL = 0

            for i, (x, _) in enumerate(loader):
                x = x.to(self.device, non_blocking=True)
                z_mean, z_logvar = self.model.forward(x)
                z = reparametrize(z_mean, z_logvar)
                decoder_mean = self.model.decoder(z)
                running_loss += self.loss(self.beta, x, z_mean, z_logvar, decoder_mean).item()
                running_recons += self.reconstruction_loss(x, decoder_mean)
                running_KL += self.KL(z_mean, z_logvar)
                if i % 1000 == 999:
                    return running_loss / 1000, running_recons / 1000, running_KL / 1000

    def train(self, train_loader, validation_loader, PATH, v=0):
        best_vloss = 1e6
        epoch_early_stop = 3

        if self.train_bool:
            early_stop = 0  ## Counts the number of epochs without improvement
            results = []
            if self.save_loss:
                train_loss, train_recons, train_kl, test_loss = self.evaluate_initialization(
                    train_loader, validation_loader
                )
                disentangling_metric = self.disentangling_metric_estimate(train_loader)
                results.append([-1, train_loss, train_recons.item(), train_kl.item(), test_loss, disentangling_metric])
            if self.save_epochs:
                torch.save(self.model, "model/model_weight_n_-1.pth")
            # Training loop
            for epoch in range(self.EPOCHS):
                if early_stop >= epoch_early_stop:
                    if self.save_loss:
                        with open(f"saved_data/avg_models/model_z{self.z_dim}_b{self.beta}_v{v}.npy", "wb") as f:
                            np.save(f, results)
                    return PATH

                self.model.train(True)
                avg_loss, avg_recons, avg_KL = self.train_one_epoch(epoch, train_loader)
                if self.save_epochs:
                    torch.save(self.model, f"model/model_weight_n_{epoch * 3 + 2}.pth")
                # Evaluation on the validation set
                self.model.train(False)
                running_vloss = 0.0
                with torch.no_grad():
                    for x, _ in validation_loader:
                        # x = x.to(self.device)
                        x = x.to(self.device, non_blocking=True)
                        z_mean, z_logvar = self.model.forward(x)
                        z = reparametrize(z_mean, z_logvar)
                        decoder_mean = self.model.decoder(z)
                        beta_vae_loss = self.loss(self.beta, x, z_mean, z_logvar, decoder_mean)
                        running_vloss += beta_vae_loss.item()

                avg_vloss = running_vloss / len(validation_loader)

                if self.save_loss:
                    disentangling_metric = self.disentangling_metric_estimate(train_loader)
                    results.append([epoch, avg_loss, avg_recons.item(), avg_KL.item(), avg_vloss, disentangling_metric])

                if avg_vloss < best_vloss:
                    early_stop = 0
                    best_vloss = avg_vloss
                    torch.save(self.model, PATH)
                    # if self.save_epochs:
                    #     torch.save(self.model, 'model/model_weight_epoch_{}.pth'.format(epoch+1))

                else:
                    early_stop += 1

        if self.save_loss:
            with open(f"saved_data/avg_models/model_z{self.z_dim}_b{self.beta}_v{v}.npy", "wb") as f:
                np.save(f, results)
        return PATH

    def evaluate_initialization(self, train_loader, test_loader):
        with torch.no_grad():
            train_loss, train_recons, train_kl = self.evaluate_loss(train_loader)
            test_loss, _, _ = self.evaluate_loss(test_loader)
        return train_loss, train_recons, train_kl, test_loss

    def disentangling_metric_estimate(self, train_loader):
        n_b = 1
        emp_std = compute_estimate_std(self.model, n_b, train_loader, self.device)
        z_dim = self.model.z_dim
        z_min = -3 * np.ones(z_dim)
        z_max = 3 * np.ones(z_dim)
        disentangling_metric, _ = metric_disentangling(self.model, z_min, z_max, emp_std.cpu(), self.std_threshold)
        return disentangling_metric


def train_batch_models(args, train_loader, test_loader):
    args.train = True
    args.z_dim = 10
    n_rep = 4  # Number of times each model is computed
    for r in range(n_rep):
        PATH = f"model/average_models/model_weights_z{args.z_dim}_b{args.beta}_v{r}.pth"
        net = Solver(args)
        net.train(train_loader, test_loader, PATH, v=r)


def train_batch_beta(args, in_size, train_loader, test_loader):
    args.train = True
    args.zdim = 2
    PATH = f"model/average_models/mouse_pup/model_b_{args.beta}.pth"
    net = Solver(args, in_size)
    net.train(train_loader, test_loader, PATH)
