import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


# 定义图卷积层
class GraphConv(nn.Linear):
    '''
    graph convolutional layers.
    ----
    inputs:

    ----
    outputs:

    '''

    def __init__(self, in_feat_dim, out_feat_dim, device, bias=True):
        super(GraphConv, self).__init__(in_feat_dim, out_feat_dim, bias=bias)
        # 设置初始值为False，使用掩码后设置为True
        self.mask_flag = False
        self.device = device

    def set_mask(self, mask):
        self.mask = mask.to(self.device)
        # 设置掩码，将权重张量乘以掩码，只保留掩码为 1 的位置的权重
        self.weight.data = self.weight.data.to(self.device) * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight.to(self.device) * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


def all_pairs_gaussian_kl(mu, sigma, eps=1e-8):
    '''

    '''
    sigma_sq = torch.square(sigma) + eps
    sigma_sq_inv = torch.reciprocal(sigma_sq)

    term1 = torch.mm(sigma_sq, torch.transpose(sigma_sq_inv, 0, 1))

    r = torch.mm(mu * mu, torch.transpose(sigma_sq_inv, 0, 1))
    r2 = mu * mu * sigma_sq_inv
    r2 = torch.sum(r2, 1)

    term2 = 2 * torch.mm(mu, torch.transpose(mu * sigma_sq_inv, 0, 1))
    term2 = r - term2 + torch.transpose(r2.view(-1, 1), 0, 1)

    r = torch.sum(torch.log(sigma_sq), 1)
    r = r.view(-1, 1)
    term3 = r - torch.transpose(r, 0, 1)

    return .5 * (term1 + term2 + term3)


def kl_conditional_and_marg(mu, log_sigma_sq, latent_dim):
    '''

    '''
    sigma = torch.exp(.5 * log_sigma_sq)
    all_pairs_gkl = all_pairs_gaussian_kl(mu, sigma) - .5 * latent_dim

    return torch.mean(all_pairs_gkl)


class INV_VAE(nn.Module):
    '''
    invariant variational auto-encoders.
    ----
    inputs:

    ----
    outputs:

    '''

    def __init__(self, config):
        super(INV_VAE, self).__init__()
        self.n_nodes = config.n_nodes
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.nuisance_dim = config.nuisance_dim
        self.n_enc_layers = config.n_enc_layers
        self.n_dec_layers = config.n_dec_layers
        self.drop_out = config.drop_out
        self.beta = config.beta
        self.gamma = config.gamma
        self.add_reg = config.add_reg
        self.y_dim = config.y_dim
        self.device = config.device

        # encoder layers (inference model)
        self.W = Variable(torch.randn(self.n_dec_layers, 1), requires_grad=True)  # add cuda() if gpu available
        self.b = Variable(torch.randn(self.latent_dim * self.latent_dim),
                          requires_grad=True)  # add cuda() if gpu available
        if 'cuda' in self.device.type:
            self.W = self.W.cuda()
            self.b = self.b.cuda()
        enc_layers = nn.ParameterList(
            [nn.Linear(self.latent_dim * self.latent_dim, self.hidden_dim) for i in range(self.n_enc_layers)]
        )
        self.enc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_drop = nn.Dropout(p=self.drop_out)
        self.encoder = nn.Sequential(*enc_layers)

        # decoder layers (generative model)
        self.dec_layers = nn.ParameterList(
            [nn.Linear(self.latent_dim + self.nuisance_dim, self.n_nodes).to(self.device) for i in
             range(self.n_dec_layers)]
        )

        # graph convolution layers
        self.gc_layers = nn.ParameterList(
            [GraphConv(self.n_nodes, self.n_nodes, config.device).to(self.device) for i in range(self.n_dec_layers)]
        )

        self.fc = nn.Linear(self.n_nodes * self.n_nodes, self.n_nodes * self.n_nodes)

        if self.add_reg:
            self.reg = nn.Linear(self.latent_dim, self.y_dim)

    def encode(self, x_input):
        output = F.relu(self.encoder(x_input))
        output = self.enc_drop(output)
        return self.enc_mu(output), self.enc_logvar(output)

    def reparameterize(self, mu, logvar):
        sd = torch.exp(.5 * logvar)
        eps = torch.randn_like(sd)
        return mu + eps * sd

    def decode(self, z_input, c_input):
        z_c_input = torch.cat((z_input, c_input), -1).to(self.device)
        dec_out = [torch.sigmoid(self.dec_layers[i](z_c_input)) for i in range(self.n_dec_layers)]
        gc_out = [torch.sigmoid(self.gc_layers[i](dec_out[i])) for i in range(self.n_dec_layers)]
        bmm_out = [torch.bmm(gc_out[i].unsqueeze(2), gc_out[i].unsqueeze(1)).view(-1, self.n_nodes * self.n_nodes, 1) \
                   for i in range(self.n_dec_layers)]
        output = torch.cat(bmm_out, 2)
        output = torch.bmm(output, self.W.expand(output.shape[0], self.n_dec_layers, 1))
        output = output.view(-1, self.n_nodes * self.n_nodes) + self.b.expand(output.shape[0],
                                                                              self.n_nodes * self.n_nodes)
        output = torch.exp(self.fc(output))
        return output

    def forward(self, x_input, c_input):
        mu, logvar = self.encode(x_input.view(-1, self.n_nodes * self.n_nodes))
        z_sample = self.reparameterize(mu, logvar)
        x_output = self.decode(z_sample, c_input)
        if self.add_reg:
            y_output = self.reg(mu)
            return x_output, y_output, mu, logvar
        else:
            return x_output, mu, logvar

    def set_mask(self, masks):
        for i in range(self.n_dec_layers):
            self.gc_layers[i].set_mask(masks[i])

    def loss(self, x_output, x_input, mu, logvar):
        nll = F.poisson_nll_loss(x_output, x_input.view(-1, self.n_nodes * self.n_nodes), reduction='sum',
                                 log_input=False)
        kl = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        inv_loss = kl_conditional_and_marg(mu, logvar, self.latent_dim)
        loss = .5 * (1 + self.gamma) * nll + self.beta * kl + self.gamma * inv_loss
        return loss, nll, kl, inv_loss

    def reg_loss(self, x_output, x_input, y_output, y_input, mu, logvar):
        nll = F.poisson_nll_loss(x_output, x_input.view(-1, self.n_nodes * self.n_nodes), reduction='sum',
                                 log_input=False)
        kl = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        inv_loss = kl_conditional_and_marg(mu, logvar, self.latent_dim)
        mse = F.mse_loss(y_output.view(-1, 1), y_input.view(-1, 1), reduction='sum')
        loss = .5 * (1 + self.gamma) * nll + self.beta * kl + self.gamma * inv_loss + mse
        # loss = self.gamma * nll + (self.beta + self.gamma) * inv_loss + mse
        return loss, nll, kl, inv_loss, mse

    def custom_train(self, epoch, train_loader, model, optimizer, device, n_epoch_display=5):
        model.train()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        tot_inv_loss = 0

        cvae_recon_mse_t = 0

        n = len(train_loader.dataset)
        for batch_idx, (batch_x, batch_c) in enumerate(train_loader):
            x_input = batch_x.to(device)
            c_input = batch_c.to(device)
            optimizer.zero_grad()
            x_output, mu, logvar = model(x_input, c_input)
            loss, nll, kl, inv_loss = model.loss(x_output, x_input, mu, logvar)
            loss.backward()

            cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, 68 * 68),
                                        reduction='sum')

            tot_loss += loss.item()
            tot_nll += nll.item()
            tot_kl += kl.item()
            tot_inv_loss += inv_loss.item()

            cvae_recon_mse_t += cvae_recon_mse.item()

            optimizer.step()
        if (epoch % n_epoch_display) == 0:
            print('epoch: {} train loss: {:.3f} nll: {:.3f} kl: {:.3f} inv_loss: {:.3f} cvae_recon_mse: '
                  '{:.3f}'.format(epoch, tot_loss / n, tot_nll / n, tot_kl / n, tot_inv_loss / n,
                                  cvae_recon_mse_t / n))
        losses = [[tot_loss / n], [tot_nll / n], [tot_kl / n], [tot_inv_loss / n],
                  [cvae_recon_mse_t / n]]
        return losses

    def custom_test(self, epoch, test_loader, model, device, n_epoch_display=5):
        model.eval()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        tot_inv_loss = 0

        cvae_recon_mse_t = 0

        n = len(test_loader.dataset)
        with torch.no_grad():
            for batch_idx, (batch_x, batch_c) in enumerate(test_loader):
                x_input = batch_x.to(device)
                c_input = batch_c.to(device)
                x_output, mu, logvar = model(x_input, c_input)
                loss, nll, kl, inv_loss = model.loss(x_output, x_input, mu, logvar)

                cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, 68 * 68),
                                            reduction='sum')

                tot_loss += loss.item()
                tot_nll += nll.item()
                tot_kl += kl.item()
                tot_inv_loss += inv_loss.item()

                cvae_recon_mse_t += cvae_recon_mse.item()

        if (epoch % n_epoch_display) == 0:
            print('epoch: {} test loss: {:.3f} nll: {:.3f} kl: {:.3f} inv_loss: {:.3f} cvae_recon_mse: '
                  '{:.3f}'.format(epoch, tot_loss / n, tot_nll / n, tot_kl / n, tot_inv_loss / n,
                                  cvae_recon_mse_t / n))
        losses = [[tot_loss / n], [tot_nll / n], [tot_kl / n], [tot_inv_loss / n],
                  [cvae_recon_mse_t / n]]
        return losses

    def reg_train(self, epoch, train_loader, model, optimizer, device, n_epoch_display=5):
        model.train()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        tot_inv_loss = 0
        tot_rmse = 0

        cvae_recon_mse_t = 0

        n = len(train_loader.dataset)
        for batch_idx, (batch_x, batch_c, batch_y) in enumerate(train_loader):
            x_input = batch_x.to(device)
            c_input = batch_c.to(device)
            y_input = batch_y.to(device)
            optimizer.zero_grad()
            x_output, y_output, mu, logvar = model(x_input, c_input)
            loss, nll, kl, inv_loss, mse = model.reg_loss(x_output, x_input, y_output, y_input, mu, logvar)
            loss.backward()

            cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, 68 * 68),
                                        reduction='sum')

            tot_loss += loss.item()
            tot_nll += nll.item()
            tot_kl += kl.item()
            tot_inv_loss += inv_loss.item()
            tot_rmse += mse.item()

            cvae_recon_mse_t += cvae_recon_mse.item()

            optimizer.step()
        if (epoch % n_epoch_display) == 0:
            print('epoch: {} train loss: {:.3f} nll: {:.3f} kl: {:.3f} inv_loss: {:.3f} rmse: {:.3f} cvae_recon_mse: '
                  '{:.3f}'.format(epoch, tot_loss / n, tot_nll / n, tot_kl / n, tot_inv_loss / n, tot_rmse / n, cvae_recon_mse_t / n))
        losses = [[tot_loss / n], [tot_nll / n], [tot_kl / n], [tot_inv_loss / n], [tot_rmse / n], [cvae_recon_mse_t / n]]

        return losses

    def reg_test(self, epoch, test_loader, model, device, n_epoch_display=5):
        model.eval()
        tot_loss = 0
        tot_nll = 0
        tot_kl = 0
        tot_inv_loss = 0
        tot_rmse = 0

        cvae_recon_mse_t = 0

        n = len(test_loader.dataset)
        with torch.no_grad():
            for batch_idx, (batch_x, batch_c, batch_y) in enumerate(test_loader):
                x_input = batch_x.to(device)
                c_input = batch_c.to(device)
                y_input = batch_y.to(device)
                x_output, y_output, mu, logvar = model(x_input, c_input)
                loss, nll, kl, inv_loss, mse = model.reg_loss(x_output, x_input, y_output, y_input, mu, logvar)

                cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, 68 * 68),
                                            reduction='sum')

                tot_loss += loss.item()
                tot_nll += nll.item()
                tot_kl += kl.item()
                tot_inv_loss += inv_loss.item()
                tot_rmse += mse.item()

                cvae_recon_mse_t += cvae_recon_mse.item()

        if (epoch % n_epoch_display) == 0:
            print('epoch: {} test loss: {:.3f} nll: {:.3f} kl: {:.3f} inv_loss: {:.3f} rmse: {:.3f} cvae_recon_mse: '
                  '{:.3f}'.format(epoch, tot_loss / n, tot_nll / n, tot_kl / n, tot_inv_loss / n, tot_rmse / n,
                                  cvae_recon_mse_t / n))
        losses = [[tot_loss / n], [tot_nll / n], [tot_kl / n], [tot_inv_loss / n], [tot_rmse / n],
                  [cvae_recon_mse_t / n]]
        return losses
