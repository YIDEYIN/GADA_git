import torch
import torch.nn.functional as F
import numpy as np
from .model.utils import get_mixed_prediction, different_p_q_objectives


def train_gada_joint(train_loader, test_loader, diffusion, dae, dae_optimizer, cvae, cvae_optimizer, num_epoch, config):
    train_losses = []
    test_losses = []

    for epo in range(num_epoch):

        cvae_recon_t = 0
        cvae_nelbo_t = 0
        cvae_mse_t = 0
        kl_t = 0
        cvae_loss_t = 0
        diffusion_loss_t = 0

        cvae_recon_mse_t = 0

        n = len(train_loader.dataset)

        dae.train()
        cvae.train()

        for i, (x_input, c_input, y_input) in enumerate(train_loader):
            x_input = x_input.to(config.device)
            c_input = c_input.to(config.device)
            y_input = y_input.to(config.device)

            dae_optimizer.zero_grad()
            cvae_optimizer.zero_grad()

            with torch.set_grad_enabled(config.train_cvae):
                x_output, y_output, z_sample, all_log_q = cvae(x_input, c_input)

                # print('x_output', x_output[:1])
                # print('y_output', y_output[:1])
                # print(y_output.shape)
                # print('z_sample', z_sample[:1])

                # 分离z使得更新cvae与diffusion时可以传递两次loss
                z = z_sample.detach()

                cvae_recon_loss = F.poisson_nll_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes)
                                                     , reduction='none', log_input=False)

                cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes),
                                            reduction='sum')

                cvae_mse = F.mse_loss(y_output.view(-1, 1), y_input.view(-1, 1), reduction='none')
                # 每个样本平均损失
                cvae_recon_loss_mean = torch.sum(cvae_recon_loss, dim=1)
                cvae_mse_mean = torch.sum(cvae_mse, dim=1)
                cvae_neg_entropy = torch.sum(all_log_q, dim=1)

            noise = torch.randn(size=z_sample.size(), device='cuda')

            t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                diffusion.iw_quantities(config.batch_size, config.time_eps, config.iw_sample_p,
                                        config.iw_subvp_like_vp_sde)
            z_t_p = diffusion.sample_q(z, noise, var_t_p, m_t_p)

            # 是否采样不同批的t训练vae与dae
            if config.iw_sample_q in ['ll_uniform', 'll_iw']:
                t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                    diffusion.iw_quantities(config.batch_size, config.time_eps, config.iw_sample_q,
                                            config.iw_subvp_like_vp_sde)
                z_t_q = diffusion.sample_q(z, noise, var_t_q, m_t_q)

                z_t_p = z_t_p.detach().requires_grad_(True)
                z_t = torch.cat([z_t_p, z_t_q], dim=0)
                c_t = torch.cat([c_input, c_input], dim=0)
                var_t = torch.cat([var_t_p, var_t_q], dim=0)
                t = torch.cat([t_p, t_q], dim=0)
                noise = torch.cat([noise, noise], dim=0)
            else:
                z_t, c_t, m_t, var_t, t, g2_t = z_t_p, c_input, m_t_p, var_t_p, t_p, g2_t_p

            # print(t.shape)

            z_t.requires_grad_(True)
            mixing_component = diffusion.mixing_component(z_t, var_t, t, enabled=dae.mixed_prediction)
            pred_score = dae(z_t, c_t, t)

            # print("mixing_component.shape", mixing_component.shape)
            # print("pred_score.shape", pred_score.shape)

            eps = get_mixed_prediction(dae.mixed_prediction, pred_score, dae.mixing_logit, mixing_component)

            # print("eps.shape", eps.shape)
            # print("noise.shape", noise.shape)

            # l2_term = torch.square(eps - noise)
            l2_term = torch.pow(eps - noise, 2)

            # print("l2_term.shape", l2_term.shape)

            # 是否采样不同批的t训练vae与dae
            if config.iw_sample_q in ['ll_uniform', 'll_iw']:
                l2_term_p, l2_term_q = torch.chunk(l2_term, chunks=2, dim=0)
                p_objective = torch.sum(obj_weight_t_p * l2_term_p, dim=1)
                cross_entropy_per_var = obj_weight_t_q * l2_term_q
            else:
                p_objective = torch.sum(obj_weight_t_p * l2_term, dim=1)
                cross_entropy_per_var = obj_weight_t_q * l2_term

            all_neg_log_p = cross_entropy_per_var + diffusion.cross_entropy_const(config.time_eps)
            diffusion_cross_entropy = torch.sum(all_neg_log_p, dim=1)

            kl_all = torch.tensor([torch.sum(neg_log_p + log_q) for log_q, neg_log_p in
                                   zip(all_log_q, all_neg_log_p)]).to(config.device)

            # print('diffusion_cross_entropy', diffusion_cross_entropy)
            # print('all_neg_log_p', all_neg_log_p)
            # print('cvae_neg_entropy', cvae_neg_entropy)

            # 计算cvae的elbo
            nelbo_loss = kl_all + cvae_recon_loss_mean
            # 加上y的loss
            cvae_loss = nelbo_loss + cvae_mse_mean
            kl = cvae_neg_entropy + diffusion_cross_entropy

            # print('cvae_loss', cvae_loss)
            # print('diffusion_loss', p_objective)
            # print('cvae_recon_loss', cvae_recon_loss_mean)
            # print('cvae_mse', cvae_mse_mean)
            # print('kl', kl)
            # print('kl_all', kl_all)
            # print('nelbo_loss = kl + ', nelbo_loss)

            q_loss = torch.mean(cvae_loss)
            p_loss = torch.mean(p_objective)

            if config.train_cvae:
                q_loss.backward(retain_graph=different_p_q_objectives(config.iw_sample_p, config.iw_sample_q))
                cvae_optimizer.step()

            if different_p_q_objectives(config.iw_sample_p, config.iw_sample_q) or not config.train_cvae:
                if config.train_cvae:
                    dae_optimizer.zero_grad()
                p_loss.backward()
            dae_optimizer.step()

            cvae_loss_t += torch.sum(cvae_loss).item()
            diffusion_loss_t += torch.sum(p_objective).item()
            cvae_recon_t += torch.sum(cvae_recon_loss_mean).item()
            cvae_mse_t += torch.sum(cvae_mse_mean).item()
            kl_t += torch.sum(kl).item()
            cvae_nelbo_t += torch.sum(nelbo_loss).item()

            cvae_recon_mse_t += cvae_recon_mse.item()

        n_epoch_display = 5

        if (epo % n_epoch_display) == 0:
            print('train epoch: {} cvae_loss: {:.3f}; diffusion_loss: {:.3f}; cvae_recon_loss: {:.3f} '
                  'cvae_mse: {:.3f} kl: {:.3f} cvae_nelbo: {:.3f} cvae_recon_mse: {:.3f}'
                  .format(epo, cvae_loss_t / n,
                          diffusion_loss_t / n, cvae_recon_t / n,
                          cvae_mse_t / n, kl_t / n,
                          cvae_nelbo_t / n, cvae_recon_mse_t / n))
        train_loss = [[cvae_loss_t / n], [diffusion_loss_t / n], [cvae_recon_t / n], [cvae_mse_t / n], [kl_t / n],
                      [cvae_nelbo_t / n], [cvae_recon_mse_t / n]]
        train_losses.append(train_loss)

        # torch.save(cvae.state_dict(), "cvae_model")
        # torch.save(dae.state_dict(), "score_model")

        with torch.set_grad_enabled(False):
            test_loss = test_gada_joint(test_loader, diffusion, dae, cvae, epo, config)
            test_losses.append(test_loss)

    return np.array(train_losses), np.array(test_losses)


def test_gada_joint(test_loader, diffusion, dae, cvae, epo, config):
    cvae_recon_t = 0
    cvae_nelbo_t = 0
    cvae_mse_t = 0
    kl_t = 0
    cvae_loss_t = 0
    diffusion_loss_t = 0

    cvae_recon_mse_t = 0

    n = len(test_loader.dataset)

    dae.eval()
    cvae.eval()

    with torch.no_grad():
        for i, (x_input, c_input, y_input) in enumerate(test_loader):
            x_input = x_input.to(config.device)
            c_input = c_input.to(config.device)
            y_input = y_input.to(config.device)

            x_output, y_output, z_sample, all_log_q = cvae(x_input, c_input)

            cvae_recon_loss = F.poisson_nll_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes)
                                                 , reduction='none', log_input=False)

            cvae_recon_mse = F.mse_loss(x_output, x_input.view(-1, config.n_nodes * config.n_nodes),
                                        reduction='sum')

            cvae_mse = F.mse_loss(y_output.view(-1, 1), y_input.view(-1, 1), reduction='none')
            # 每个样本平均损失
            cvae_recon_loss_mean = torch.sum(cvae_recon_loss, dim=1)
            cvae_mse_mean = torch.sum(cvae_mse, dim=1)
            cvae_neg_entropy = torch.sum(all_log_q, dim=1)

            noise = torch.randn(size=z_sample.size(), device='cuda')

            t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                diffusion.iw_quantities(config.batch_size, config.time_eps, config.iw_sample_p,
                                        config.iw_subvp_like_vp_sde)
            z_t_p = diffusion.sample_q(z_sample, noise, var_t_p, m_t_p)

            # 是否采样不同批的t训练vae与dae
            if config.iw_sample_q in ['ll_uniform', 'll_iw']:
                t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                    diffusion.iw_quantities(config.batch_size, config.time_eps, config.iw_sample_q,
                                            config.iw_subvp_like_vp_sde)
                z_t_q = diffusion.sample_q(z_sample, noise, var_t_q, m_t_q)

                z_t_p = z_t_p.detach().requires_grad_(True)
                z_t = torch.cat([z_t_p, z_t_q], dim=0)
                c_t = torch.cat([c_input, c_input], dim=0)
                var_t = torch.cat([var_t_p, var_t_q], dim=0)
                t = torch.cat([t_p, t_q], dim=0)
                noise = torch.cat([noise, noise], dim=0)
            else:
                z_t, c_t, m_t, var_t, t, g2_t = z_t_p, c_input, m_t_p, var_t_p, t_p, g2_t_p

            mixing_component = diffusion.mixing_component(z_t, var_t, t, enabled=dae.mixed_prediction)
            pred_score = dae(z_t, c_t, t)

            eps = get_mixed_prediction(dae.mixed_prediction, pred_score, dae.mixing_logit, mixing_component)

            l2_term = torch.pow(eps - noise, 2)

            # 是否采样不同批的t测试vae与dae
            if config.iw_sample_q in ['ll_uniform', 'll_iw']:
                l2_term_p, l2_term_q = torch.chunk(l2_term, chunks=2, dim=0)
                p_objective = torch.sum(obj_weight_t_p * l2_term_p, dim=1)
                cross_entropy_per_var = obj_weight_t_q * l2_term_q
            else:
                p_objective = torch.sum(obj_weight_t_p * l2_term, dim=1)
                cross_entropy_per_var = obj_weight_t_q * l2_term

            all_neg_log_p = cross_entropy_per_var + diffusion.cross_entropy_const(config.time_eps)
            diffusion_cross_entropy = torch.sum(all_neg_log_p, dim=1)

            kl_all = torch.tensor([torch.sum(neg_log_p + log_q) for log_q, neg_log_p in
                                   zip(all_log_q, all_neg_log_p)]).to(config.device)

            # 计算cvae的elbo
            nelbo_loss = kl_all + cvae_recon_loss_mean
            # 加上y的loss
            cvae_loss = nelbo_loss + cvae_mse_mean
            kl = cvae_neg_entropy + diffusion_cross_entropy

            cvae_loss_t += torch.sum(cvae_loss).item()
            diffusion_loss_t += torch.sum(p_objective).item()
            cvae_recon_t += torch.sum(cvae_recon_loss_mean).item()
            cvae_mse_t += torch.sum(cvae_mse_mean).item()
            kl_t += torch.sum(kl).item()
            cvae_nelbo_t += torch.sum(nelbo_loss).item()

            cvae_recon_mse_t += cvae_recon_mse.item()

    n_epoch_display = 5

    if (epo % n_epoch_display) == 0:
        print('test epoch: {} cvae_loss: {:.3f}; diffusion_loss: {:.3f}; cvae_recon_loss: {:.3f} '
              'cvae_mse: {:.3f} kl: {:.3f} cvae_nelbo: {:.3f} cvae_recon_mse: {:.3f}'
              .format(epo, cvae_loss_t / n,
                      diffusion_loss_t / n, cvae_recon_t / n,
                      cvae_mse_t / n, kl_t / n,
                      cvae_nelbo_t / n, cvae_recon_mse_t / n))
    test_loss = [[cvae_loss_t / n], [diffusion_loss_t / n], [cvae_recon_t / n], [cvae_mse_t / n], [kl_t / n],
                 [cvae_nelbo_t / n], [cvae_recon_mse_t / n]]

    return np.array(test_loss)
