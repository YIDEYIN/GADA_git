import torch
from torch import nn
from .utils import init_temb_fun, mask_inactive_variables


# score模型
class score_sde(nn.Module):
    def __init__(self, config):
        super(score_sde, self).__init__()
        self.latent_dim = config.latent_dim
        self.hidden1_dim = config.hidden1_dim
        self.hidden2_dim = config.hidden2_dim
        self.embedding_type = config.embedding_type
        self.embedding_scale = config.embedding_scale
        self.embedding_dim = config.embedding_dim
        self.group_dim = config.group_dim
        self.drop_out = config.drop_out
        self.uncond_prob = config.uncond_prob
        self.device = config.device

        # 是否启用混合预测
        self.mixed_prediction = config.mixed_prediction
        if self.mixed_prediction:
            init = config.mixing_logit_init * torch.ones(size=[1, 1])
            self.mixing_logit = torch.nn.Parameter(init, requires_grad=True)
            self.is_active = None
        else:
            self.mixing_logit = None
            self.is_active = None

        # Diffuision
        self.t_embedding = init_temb_fun(self.embedding_type, self.embedding_scale, self.embedding_dim)

        self.score = nn.Sequential(nn.Linear(self.latent_dim + self.embedding_dim + self.group_dim, self.hidden1_dim),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_out),
                                   nn.Linear(self.hidden1_dim, self.hidden2_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden2_dim, self.latent_dim))

    # diffusion-SGM
    def sample_q(self, z_init, noise, var_t, m_t):
        """ returns a sample from diffusion process at time t """
        return m_t * z_init + torch.sqrt(var_t) * noise

    # 如果没有采样噪声
    def sde_noise(self, z_init, var_t, m_t):
        noise = torch.randn(z_init.shape)
        z_noise = m_t * z_init + torch.sqrt(var_t) * noise
        return z_noise

    def forward(self, z_noise, t, c_input):
        t = torch.squeeze(t)
        t = self.t_embedding(t)
        # print(t.shape)

        # mask out inactive variables
        if self.mixed_prediction and self.is_active is not None:
            z_noise = mask_inactive_variables(z_noise, self.is_active)

        # print("z_noise.shape", z_noise.shape)
        # print("c_input.shape", c_input.shape)
        # print("t.shape", t.shape)

        # 创建随即掩码，将一些组别信息设为空，训练无条件网络
        context_mask = torch.bernoulli(torch.zeros_like(c_input) + self.uncond_prob).to(self.device)
        # 互换0，1
        context_mask = 1 - context_mask
        c_input = c_input * context_mask

        z_c_t_noise = torch.cat((z_noise, t, c_input), -1).to(self.device)
        epsilon = self.score(z_c_t_noise).to(self.device)
        return epsilon
