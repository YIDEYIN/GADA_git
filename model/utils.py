import math
import torch
from torch import nn
import numpy as np


# 是否使用混合预测
def get_mixed_prediction(mixed_prediction, param, mixing_logit, mixing_component=None):
    if mixed_prediction:
        assert mixing_component is not None, 'Provide mixing component when mixed_prediction is enabled.'
        coeff = torch.sigmoid(mixing_logit)
        param = (1 - coeff) * mixing_component + coeff * param

    return param


# 计算Jacobian矩阵的迹
def trace_df_dx_hutchinson(f, x, noise, no_autograd):
    if no_autograd:
        # the following is compatible with checkpointing
        torch.sum(f * noise).backward()
        # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
        jvp = x.grad
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        x.grad = None
    else:
        jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

    return trJ


# 计算标准正态分布的对数概率密度
def log_p_standard_normal(samples):
    log_p = - 0.5 * torch.square(samples) - 0.5 * np.log(2 * np.pi)
    return log_p


# 计算带有指定方差的对数概率密度
def log_p_var_normal(samples, var):
    log_p = - 0.5 * torch.square(samples) / var - 0.5 * np.log(var) - 0.5 * np.log(2 * np.pi)
    return log_p


# mask不活跃的变量
def mask_inactive_variables(x, is_active):
    x = x * is_active
    return x


# 判断p与q是否相同
def different_p_q_objectives(iw_sample_p, iw_sample_q):
    assert iw_sample_p in ['ll_uniform', 'drop_all_uniform', 'll_iw', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw',
                           'drop_sigma2t_uniform']
    assert iw_sample_q in ['reweight_p_samples', 'll_uniform', 'll_iw']
    # In these cases, we reuse the likelihood-based p-objective (either the uniform sampling version or the importance
    # sampling version) also for q.
    if iw_sample_p in ['ll_uniform', 'll_iw'] and iw_sample_q == 'reweight_p_samples':
        return False
    # In these cases, we are using a non-likelihood-based objective for p, and hence definitly need to use another q
    # objective.
    else:
        return True


# 权重重置
def reset_weights(model):
    '''
    try resetting model weights to avoid weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# 获得t的embedding函数
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.scale
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class RandomFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(RandomFourierEmbedding, self).__init__()
        self.w = nn.Parameter(torch.randn(size=(1, embedding_dim // 2)) * scale, requires_grad=False)

    def forward(self, timesteps):
        emb = torch.matmul(timesteps, self.w * 2 * 3.14159265359)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


def init_temb_fun(embedding_type, embedding_scale, embedding_dim):
    if embedding_type == 'positional':
        temb_fun = PositionalEmbedding(embedding_dim, embedding_scale)
    elif embedding_type == 'fourier':
        temb_fun = RandomFourierEmbedding(embedding_dim, embedding_scale)
    else:
        raise NotImplementedError

    return temb_fun


# 启用kl退火
def kl_coefficient(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
    # return max(min(max_kl_coeff * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff),
               min_kl_coeff)
