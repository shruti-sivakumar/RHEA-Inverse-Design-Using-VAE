import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Helper: simple MLP
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act=nn.GELU, ln=False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act()]
        if ln:
            layers.append(nn.LayerNorm(hidden))
        layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------
# Conditional Prior p(z|y_cond)
# -----------------------
class ConditionalPrior(nn.Module):
    def __init__(self, y_dim, z_dim, hidden=128):
        super().__init__()
        self.to_mu     = MLP(y_dim, hidden, z_dim)
        self.to_logvar = MLP(y_dim, hidden, z_dim)

    def forward(self, y):
        return self.to_mu(y), self.to_logvar(y)


# -----------------------
# Encoder q(z|x, y_cond)
# -----------------------
class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden=256):
        super().__init__()
        self.net_mu     = MLP(x_dim + y_dim, hidden, z_dim, ln=True)
        self.net_logvar = MLP(x_dim + y_dim, hidden, z_dim, ln=True)

    def forward(self, x, y_cond):
        xy = torch.cat([x, y_cond], dim=-1)
        return self.net_mu(xy), self.net_logvar(xy)


# -----------------------
# Decoder p(x|z, y_cond)
# -----------------------
class DecoderStrongCond(nn.Module):
    """Decoder that concatenates y_cond at every layer (strong conditioning)."""
    def __init__(self, x_dim, y_dim, z_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + y_dim, hidden)
        self.fc2 = nn.Linear(hidden + y_dim, hidden)
        self.fc3 = nn.Linear(hidden + y_dim, x_dim)
        self.act = nn.GELU()

    def forward(self, z, y_cond):
        h = torch.cat([z, y_cond], dim=-1)
        h = self.act(self.fc1(h))
        h = torch.cat([h, y_cond], dim=-1)
        h = self.act(self.fc2(h))
        h = torch.cat([h, y_cond], dim=-1)
        x_hat = self.fc3(h)
        return x_hat


# -----------------------
# Property Head: predict y_prop from z (+ y_cond)
# -----------------------
class PropertyHead(nn.Module):
    """Predict property y_prop from latent z (optionally conditioned on y_cond)."""
    def __init__(self, z_dim, y_cond_dim, y_prop_dim, hidden=128):
        super().__init__()
        in_dim = z_dim + y_cond_dim  # <-- changed: include y_cond
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, y_prop_dim)
        )

    def forward(self, z, y_cond):
        zy = torch.cat([z, y_cond], dim=-1)
        return self.net(zy)


# -----------------------
# CVAE
# -----------------------
class CVAE(nn.Module):
    def __init__(self, x_dim, y_cond_dim, y_prop_dim, z_dim=8, hidden=256):
        super().__init__()
        self.enc   = Encoder(x_dim, y_cond_dim, z_dim, hidden)
        self.dec   = DecoderStrongCond(x_dim, y_cond_dim, z_dim, hidden)
        self.prior = ConditionalPrior(y_cond_dim, z_dim, hidden=128)
        self.prop_head = PropertyHead(z_dim, y_cond_dim, y_prop_dim, hidden=128)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_cond):
        # Encoder
        q_mu, q_logvar = self.enc(x, y_cond)
        z = self.reparameterize(q_mu, q_logvar)

        # Prior
        p_mu, p_logvar = self.prior(y_cond)

        # Decoder
        x_hat = self.dec(z, y_cond)

        # Property prediction (now conditioned on y_cond)
        y_prop_pred = self.prop_head(z, y_cond)

        return x_hat, y_prop_pred, (q_mu, q_logvar, p_mu, p_logvar)

    def sample_prior(self, y_cond, n=1):
        """Sample latent z from conditional prior."""
        p_mu, p_logvar = self.prior(y_cond)
        std = torch.exp(0.5 * p_logvar)
        eps = torch.randn((n, p_mu.size(-1)), device=p_mu.device)
        z = p_mu + eps * std
        return z


# -----------------------
# KL Divergence
# -----------------------
def kl_divergence(q_mu, q_logvar, p_mu, p_logvar):
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    term1 = (p_logvar - q_logvar)
    term2 = (q_var + (q_mu - p_mu).pow(2)) / p_var
    kl = 0.5 * (term1 + term2 - 1.0)
    return kl.sum(dim=1).mean()
