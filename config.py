
import torch
from types import SimpleNamespace


def BIN_config_GADA():
    config = SimpleNamespace()
    config.batch_size = 16
    config.n_nodes = 68
    config.n_epochs = 100
    config.lr = 3e-6

    # cvae
    config.latent_dim = 68
    config.hidden1_dim = 512
    config.hidden2_dim = 1024
    config.group_dim = 1
    config.n_dec_layers = 5
    config.n_neighbors = 32
    config.drop_out = 0.1
    config.y_dim = 1
    config.alpha = 1.0
    config.beta = 10.0
    config.gamma = 0.1

    # diffusion
    config.mixed_prediction = True
    config.sigma2_0 = 0
    config.sigma2_min = 1e-4
    config.sigma2_max = 0.99
    config.sde_type = ['geometric_sde', 'vpsde', 'sub_vpsde', 'vesde']
    config.train_ode_eps = 1e-2
    config.train_ode_solver_tol = 1e-4
    config.time_eps = 1e-2
    config.mixing_logit_init = -3
    config.iw_sample_p = ['ll_uniform', 'll_iw', 'drop_all_uniform', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw',
                          'drop_sigma2t_uniform']
    config.iw_sample_q = ['reweight_p_samples', 'll_uniform', 'll_iw']
    config.beta_start = 0.1
    config.beta_end = 20.0

    config.cont_kl_anneal = True
    config.kl_anneal_portion_vada = 0.1
    config.kl_const_portion_vada = 0.0
    config.kl_const_coeff_vada = 0.7
    config.kl_max_coeff_vada = 1.0

    # score_sde
    config.embedding_type = ['positional', 'fourier']
    config.embedding_scale = 1
    config.embedding_dim = 32
    config.uncond_prob = 0.2
    config.guide_omega = 0.5

    # training
    config.train_cvae = True
    config.iw_subvp_like_vp_sde = True

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.device = DEVICE

    return config
