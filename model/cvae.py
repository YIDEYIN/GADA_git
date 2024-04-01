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
        if self.mask_flag:
            weight = self.weight.to(self.device) * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


# CVAE模型
class CVAE(nn.Module):
    def __init__(self, config):
        super(CVAE, self).__init__()
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
        # self.encoder = nn.Sequential(nn.Linear(self.n_nodes * self.n_nodes, self.hidden2_dim),
        #                              nn.ReLU(),
        #                              nn.Dropout(self.drop_out),
        #                              )

        # 两层
        self.encoder = nn.Sequential(nn.Linear(self.n_nodes * self.n_nodes, self.hidden2_dim),
                                     nn.ReLU(),
                                     nn.Dropout(self.drop_out),
                                     nn.Linear(self.hidden2_dim, self.hidden1_dim),
                                     nn.ReLU())

        # 四层
        # self.encoder = nn.Sequential(nn.Linear(self.n_nodes * self.n_nodes, self.hidden1_dim),
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
        self.enc_mu_0 = nn.Linear(self.hidden1_dim, self.latent_dim)
        self.enc_logvar_0 = nn.Linear(self.hidden1_dim, self.latent_dim)

        self.enc_mu_1 = nn.Linear(self.hidden1_dim, self.latent_dim)
        self.enc_logvar_1 = nn.Linear(self.hidden1_dim, self.latent_dim)

        # Decoder网络
        # 线性层
        self.dec_layers = nn.ParameterList(
            [nn.Linear(self.latent_dim, self.n_nodes).to(self.device) for i in range(self.n_dec_layers)]
        )
        # 按组别的GCN层
        self.gc_layers0 = nn.ParameterList(
            [GraphConv(self.n_nodes, self.n_nodes, config.device).to(self.device) for i in range(self.n_dec_layers)]
        )
        self.gc_layers1 = nn.ParameterList(
            [GraphConv(self.n_nodes, self.n_nodes, config.device).to(self.device) for i in range(self.n_dec_layers)]
        )
        # 全连接层
        self.fc = nn.Linear(self.n_nodes * self.n_nodes, self.n_nodes * self.n_nodes)

        # 线性回归预测y
        self.reg = nn.Linear(self.latent_dim + self.group_dim, self.y_dim)

        # 初始化参数（W为decoder中的alpha，b为控制大脑区域之间连接强度的基线参数gamma）
        self.W_0 = nn.Parameter(torch.randn(self.n_dec_layers, 1), requires_grad=True)
        self.b_0 = nn.Parameter(torch.randn(self.latent_dim * self.latent_dim), requires_grad=True)
        if 'cuda' in self.device.type:
            self.W_0 = nn.Parameter(self.W_0.cuda())
            self.b_0 = nn.Parameter(self.b_0.cuda())

        self.W_1 = nn.Parameter(torch.randn(self.n_dec_layers, 1), requires_grad=True)
        self.b_1 = nn.Parameter(torch.randn(self.latent_dim * self.latent_dim), requires_grad=True)
        if 'cuda' in self.device.type:
            self.W_1 = nn.Parameter(self.W_1.cuda())
            self.b_1 = nn.Parameter(self.b_1.cuda())

    # encoder
    def encode(self, x_input, c_input):
        x_input = x_input.view(-1, self.n_nodes * self.n_nodes).to(self.device)
        out = self.encoder(x_input)

        # 删去组别c_input中维度为 1 的维度，使其形状与out的形状匹配
        c_input = c_input.squeeze().to(self.device)

        # 根据组别信息先分成两组再进入不同的网络
        out_group_0 = out[c_input == 0]
        out_group_1 = out[c_input == 1]

        # 保留原始组别索引
        indices_0 = torch.arange(c_input.size(0), device=self.device)[c_input == 0].to(self.device)
        indices_1 = torch.arange(c_input.size(0), device=self.device)[c_input == 1].to(self.device)

        mu_0 = self.enc_mu_0(out_group_0)
        logvar_0 = self.enc_logvar_0(out_group_0)

        mu_1 = self.enc_mu_1(out_group_1)
        logvar_1 = self.enc_logvar_1(out_group_1)

        # 按原来顺序重新组合在一起
        mu_shape = torch.cat([mu_0, mu_1], dim=0)
        mu = torch.empty_like(mu_shape).to(self.device)
        mu[indices_0] = mu_0
        mu[indices_1] = mu_1

        logvar_shape = torch.cat([logvar_0, logvar_1], dim=0)
        logvar = torch.empty_like(logvar_shape).to(self.device)
        logvar[indices_0] = logvar_0
        logvar[indices_1] = logvar_1

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
            self.gc_layers0[i].set_mask(masks[i])
            self.gc_layers1[i].set_mask(masks[i])

    # decoder
    def decode(self, z_input, c_input):
        z_input = z_input.to(self.device)
        dec_out = [torch.sigmoid(self.dec_layers[i](z_input)) for i in range(self.n_dec_layers)]

        # 删去组别c_input中维度为 1 的维度，使其形状与out的形状匹配
        c_input = c_input.squeeze().to(self.device)

        # 在神经网络列表中为数据分组
        dec_out_0 = []
        dec_out_1 = []
        for i in range(len(dec_out)):
            dec_out_0.append(dec_out[i][c_input == 0])
            dec_out_1.append(dec_out[i][c_input == 1])

        # 保留原始组别索引
        indices_0 = torch.arange(c_input.size(0), device=self.device)[c_input.cpu() == 0].to(self.device)
        indices_1 = torch.arange(c_input.size(0), device=self.device)[c_input.cpu() == 1].to(self.device)

        # group = 0
        gc_out_0 = [torch.sigmoid(self.gc_layers0[i](dec_out_0[i])) for i in range(self.n_dec_layers)]
        bmm_out_0 = [
            torch.bmm(gc_out_0[i].unsqueeze(2), gc_out_0[i].unsqueeze(1)).view(-1, self.n_nodes * self.n_nodes, 1) \
            for i in range(self.n_dec_layers)]
        output_0 = torch.cat(bmm_out_0, 2)
        output_0 = torch.bmm(output_0, self.W_0.expand(output_0.shape[0], self.n_dec_layers, 1))
        output_0 = output_0.view(-1, self.n_nodes * self.n_nodes) + self.b_0.expand(output_0.shape[0],
                                                                                    self.n_nodes * self.n_nodes)
        output_0 = torch.exp(self.fc(output_0))

        # group = 1
        gc_out_1 = [torch.sigmoid(self.gc_layers1[i](dec_out_1[i])) for i in range(self.n_dec_layers)]
        bmm_out_1 = [
            torch.bmm(gc_out_1[i].unsqueeze(2), gc_out_1[i].unsqueeze(1)).view(-1, self.n_nodes * self.n_nodes, 1) \
            for i in range(self.n_dec_layers)]
        output_1 = torch.cat(bmm_out_1, 2)
        output_1 = torch.bmm(output_1, self.W_1.expand(output_1.shape[0], self.n_dec_layers, 1))
        output_1 = output_1.view(-1, self.n_nodes * self.n_nodes) + self.b_1.expand(output_1.shape[0],
                                                                                    self.n_nodes * self.n_nodes)
        output_1 = torch.exp(self.fc(output_1))

        # 按原来顺序重新组合在一起
        output_shape = torch.cat([output_0, output_1], dim=0)
        output = torch.empty_like(output_shape).to(self.device)
        output[indices_0] = output_0
        output[indices_1] = output_1

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
