# cvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act=nn.GELU, ln=False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act()]
        if ln: layers.append(nn.LayerNorm(hidden))
        layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ConditionalPrior(nn.Module):
    """ p(z|y): maps y -> (mu_y, logvar_y) """
    def __init__(self, y_dim, z_dim, hidden=64):
        super().__init__()
        self.to_mu     = MLP(y_dim, hidden, z_dim)
        self.to_logvar = MLP(y_dim, hidden, z_dim)
    def forward(self, y):
        return self.to_mu(y), self.to_logvar(y)

class Encoder(nn.Module):
    """ q(z|x,y) -> (mu, logvar) """
    def __init__(self, x_dim, y_dim, z_dim, hidden=128):
        super().__init__()
        self.net_mu     = MLP(x_dim+y_dim, hidden, z_dim, ln=True)
        self.net_logvar = MLP(x_dim+y_dim, hidden, z_dim, ln=True)
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        return self.net_mu(xy), self.net_logvar(xy)

class Decoder(nn.Module):
    """ p(x|z,y) -> x_hat """
    def __init__(self, x_dim, y_dim, z_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim+y_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, x_dim)
        )
    def forward(self, z, y):
        zy = torch.cat([z, y], dim=-1)
        return self.net(zy)

class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim=6, hidden=128):
        super().__init__()
        self.enc = Encoder(x_dim, y_dim, z_dim, hidden)
        self.dec = Decoder(x_dim, y_dim, z_dim, hidden)
        self.prior = ConditionalPrior(y_dim, z_dim, hidden=64)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        # Encoder
        q_mu, q_logvar = self.enc(x, y)
        z = self.reparameterize(q_mu, q_logvar)
        # Conditional prior
        p_mu, p_logvar = self.prior(y)
        # Decoder
        x_hat = self.dec(z, y)
        return x_hat, (q_mu, q_logvar, p_mu, p_logvar)

def kl_divergence(q_mu, q_logvar, p_mu, p_logvar):
    # KL[N(q_mu, q_var) || N(p_mu, p_var)]
    # 0.5 * [ log(p_var/q_var) + (q_var + (q_mu-p_mu)^2)/p_var - 1 ]
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    term1 = (p_logvar - q_logvar)
    term2 = (q_var + (q_mu - p_mu).pow(2)) / p_var
    kl = 0.5 * (term1 + term2 - 1.0)
    return kl.sum(dim=1).mean()  # mean over batch