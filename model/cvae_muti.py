import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


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
        if self.mask_flag == True:
            weight = self.weight.to(self.device) * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


# CVAE模型
class CVAE_MUTI(nn.Module):
    def __init__(self, config):
        super(CVAE_MUTI, self).__init__()
        self.n_nodes = config.n_nodes
        self.latent_dim = config.latent_dim
        self.hidden1_dim = config.hidden1_dim
        self.hidden2_dim = config.hidden2_dim
        self.group_dim = config.group_dim
        self.n_dec_layers = config.n_dec_layers
        self.drop_out = config.drop_out

        self.y_dim = config.y_dim
        self.device = config.device

        # Encoder网络
        # 公共网络
        # 一层
        # self.encoder = nn.Sequential(nn.Linear(self.n_nodes * self.n_nodes + self.group_dim, self.hidden2_dim),
        #                              nn.ReLU(),
        #                              nn.Dropout(self.drop_out),
        #                              )

        # 两层
        self.encoder = nn.Sequential(nn.Linear(self.n_nodes * self.n_nodes + self.group_dim, self.hidden2_dim),
                                     nn.ReLU(),
                                     nn.Dropout(self.drop_out),
                                     nn.Linear(self.hidden2_dim, self.hidden1_dim),
                                     nn.ReLU())

        # 四层
        # self.encoder = nn.Sequential(nn.Linear(self.n_nodes * self.n_nodes + self.group_dim, self.hidden1_dim),
        #                              nn.ReLU(),
        #                              nn.Dropout(self.drop_out),
        #                              nn.Linear(self.hidden1_dim, self.hidden1_dim),
        #                              nn.ReLU(),
        #                              nn.Dropout(self.drop_out),
        #                              nn.Linear(self.hidden1_dim, self.hidden2_dim),
        #                              nn.ReLU(),
        #                              nn.Dropout(self.drop_out),
        #                              nn.Linear(self.hidden2_dim, self.hidden1_dim),
        #                              nn.ReLU())

        # 分组计算均值和对数方差
        self.enc_mu = nn.Linear(self.hidden1_dim, self.latent_dim)
        self.enc_logvar = nn.Linear(self.hidden1_dim, self.latent_dim)

        # Decoder网络
        # 线性层
        self.dec_layers = nn.ParameterList(
            [nn.Linear(self.latent_dim + self.group_dim, self.n_nodes).to(self.device) for i in range(self.n_dec_layers)]
        )
        # 按组别的GCN层
        self.gc_layers = nn.ParameterList(
            [GraphConv(self.n_nodes, self.n_nodes, config.device).to(self.device) for i in range(self.n_dec_layers)]
        )
        # 全连接层
        self.fc = nn.Linear(self.n_nodes * self.n_nodes, self.n_nodes * self.n_nodes)

        # 线性回归预测y
        self.reg = nn.Linear(self.latent_dim + self.group_dim, self.y_dim)

        # 初始化参数（W为decoder中的alpha，b为控制大脑区域之间连接强度的基线参数gamma）
        self.W = nn.Parameter(torch.randn(self.n_dec_layers, 1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.latent_dim * self.latent_dim), requires_grad=True)
        if 'cuda' in self.device.type:
            self.W = nn.Parameter(self.W.cuda())
            self.b = nn.Parameter(self.b.cuda())

    # encoder
    def encode(self, x_input, c_input):
        x_input = x_input.view(-1, self.n_nodes * self.n_nodes).to(self.device)
        x_c_input = torch.cat((x_input, c_input), -1).to(self.device)
        out = self.encoder(x_c_input)
        mu = self.enc_mu(out)
        logvar = self.enc_logvar(out)
        return mu, logvar

    # 重参数得到embedding
    def reparameterize(self, mu, logvar):
        sd = torch.exp(.5 * logvar)
        eps = torch.randn_like(sd)
        z = mu + eps * sd
        return z

    # 设置掩码
    def set_mask(self, masks):
        for i in range(self.n_dec_layers):
            self.gc_layers[i].set_mask(masks[i])

    # decoder
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

    def predict(self, z_input, c_input):
        z_c_input = torch.cat((z_input, c_input), -1).to(self.device)
        y = self.reg(z_c_input)
        return y

    def forward(self, x_input, c_input):
        mu, logvar = self.encode(x_input, c_input)
        z_sample = self.reparameterize(mu, logvar)

        # 计算所有潜在变量的对数概率密度函数值
        all_log_q = []
        for i in range(z_sample.size(0)):
            dist = Normal(mu[i], logvar[i].exp().sqrt())
            log_q = dist.log_prob(z_sample[i])
            all_log_q.append(log_q)
        all_log_q = torch.stack(all_log_q)

        x_output = self.decode(z_sample, c_input)
        y_output = self.predict(z_sample, c_input)

        return x_output, y_output, z_sample, all_log_q
