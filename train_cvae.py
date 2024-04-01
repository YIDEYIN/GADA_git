import numpy as np
import torch
import torch.nn.functional as F


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


def train_cvae_independent(train_loader, test_loader, cvae, cvae_optimizer, device, num_epoch, config):
    train_losses = []
    test_losses = []

    for epo in range(num_epoch):
        cvae_loss_t = 0
        cvae_recon_loss_t = 0
        kl_t = 0
        cvae_mse_t = 0
        nelbo_loss_t = 0

        cvae_recon_mse_t = 0

        n = len(train_loader.dataset)

        cvae.train()

        for i, (x_input, c_input, y_input) in enumerate(train_loader):
            x_input = x_input.to(device)
            c_input = c_input.to(device)
            y_input = y_input.to(device)

            cvae_optimizer.zero_grad()

            x_output, y_output, z_sample, _, = cvae(x_input, c_input)
            mu, logvar = cvae.encode(x_input, c_input)

            cvae_recon_loss = F.poisson_nll_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes)
                                                 , reduction='none', log_input=False)

            cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes),
                                        reduction='sum')

            cvae_mse = F.mse_loss(y_output.view(-1, 1), y_input.view(-1, 1), reduction='none')
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            # inv_loss = torch.mean(kl_conditional_and_marg(mu, logvar, config.latent_dim), dim=1)

            # 计算每个样本的损失
            cvae_recon_loss_mean = torch.sum(cvae_recon_loss, dim=1)
            cvae_mse_mean = torch.sum(cvae_mse, dim=1)

            nelbo_loss = (config.gamma + config.alpha) * kl + config.gamma * cvae_recon_loss_mean
            cvae_loss = nelbo_loss + cvae_mse_mean

            # 计算批次每个样本的平均损失
            q_loss = torch.mean(cvae_loss)

            q_loss.backward()
            cvae_optimizer.step()

            cvae_loss_t += torch.sum(cvae_loss).item()
            cvae_recon_loss_t += torch.sum(cvae_recon_loss_mean).item()
            cvae_mse_t += torch.sum(cvae_mse_mean).item()
            kl_t += torch.sum(kl).item()
            nelbo_loss_t += torch.sum(nelbo_loss).item()

            cvae_recon_mse_t += cvae_recon_mse.item()

        n_epoch_display = 5

        if (epo % n_epoch_display) == 0:
            print(
                'train epoch: {} cvae_loss: {:.3f} cvae_recon_loss: {:.3f} cvae_mse: {:.3f} kl: {:.3f} cvae_nelbo: '
                '{:.3f} cvae_recon_mse: {:.3f}'.format(epo, cvae_loss_t / n, cvae_recon_loss_t / n, cvae_mse_t / n,
                                                       kl_t / n, nelbo_loss_t / n, cvae_recon_mse_t / n))
        train_loss = [[cvae_loss_t / n], [cvae_recon_loss_t / n], [cvae_mse_t / n], [kl_t / n], [nelbo_loss_t / n],
                      [cvae_recon_mse_t / n]]
        train_losses.append(train_loss)

        torch.save(cvae.state_dict(), "cvae_independent_model")

        with torch.set_grad_enabled(False):
            test_loss = test_cvae_independent(test_loader, cvae, device, epo, config)
            test_losses.append(test_loss)
    return np.array(train_losses), np.array(test_losses)


def test_cvae_independent(test_loader, cvae, device, epo, config):
    cvae_loss_t = 0
    cvae_recon_loss_t = 0
    kl_t = 0
    cvae_mse_t = 0
    nelbo_loss_t = 0

    cvae_recon_mse_t = 0

    n = len(test_loader.dataset)

    cvae.eval()

    with torch.no_grad():
        for i, (x_input, c_input, y_input) in enumerate(test_loader):
            x_input = x_input.to(device)
            c_input = c_input.to(device)
            y_input = y_input.to(device)

            x_output, y_output, z_sample, _, = cvae(x_input, c_input)
            mu, logvar = cvae.encode(x_input, c_input)

            cvae_recon_loss = F.poisson_nll_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes)
                                                 , reduction='none', log_input=False)

            cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes),
                                        reduction='sum')

            cvae_mse = F.mse_loss(y_output.view(-1, 1), y_input.view(-1, 1), reduction='none')
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # 计算每个样本的损失
            cvae_recon_loss_mean = torch.sum(cvae_recon_loss, dim=1)
            cvae_mse_mean = torch.sum(cvae_mse, dim=1)

            nelbo_loss = kl + cvae_recon_loss_mean
            cvae_loss = nelbo_loss + cvae_mse_mean

            cvae_loss_t += torch.sum(cvae_loss).item()
            cvae_recon_loss_t += torch.sum(cvae_recon_loss_mean).item()
            cvae_mse_t += torch.sum(cvae_mse_mean).item()
            kl_t += torch.sum(kl).item()
            nelbo_loss_t += torch.sum(nelbo_loss).item()

            cvae_recon_mse_t += cvae_recon_mse.item()

    n_epoch_display = 5

    if (epo % n_epoch_display) == 0:
        print('test epoch: {} cvae_loss: {:.3f} cvae_recon_loss: {:.3f} cvae_mse: {:.3f} kl: {:.3f} cvae_nelbo: {:.3f} '
              'cvae_recon_mse: {:.3f}'.format(epo, cvae_loss_t / n, cvae_recon_loss_t / n, cvae_mse_t / n,
                                              kl_t / n, nelbo_loss_t / n, cvae_recon_mse_t / n))
    test_loss = [[cvae_loss_t / n], [cvae_recon_loss_t / n], [cvae_mse_t / n], [kl_t / n], [nelbo_loss_t / n],
                 [cvae_recon_mse_t / n]]

    return np.array(test_loss)
