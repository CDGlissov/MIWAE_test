
import torch.nn as nn
from utils import Flatten, UnFlatten, majority_vote
import torch.distributions as td
import torch
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")

class VAE(nn.Module):
    def __init__(self, z_dim=20, bs=200, encoder_type='gaussian'):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert encoder_type in ['gaussian','studentt']
        self.encoder_type = encoder_type
        self.bs = bs

        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 4 * 4, 3 * z_dim))

        self.dec = nn.Sequential(
            nn.Linear(z_dim, 32 * 7 * 7), nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

    def elbo(self, x, K=50, beta=1):
        out_encoder = self.enc(x)
        mu_enc = out_encoder[..., :self.z_dim]
        std_enc = torch.nn.Softplus()(out_encoder[..., self.z_dim:(2 * self.z_dim)]) + 0.0001
        df_enc = torch.nn.Softplus()(out_encoder[..., (2 * self.z_dim):(3 * self.z_dim)]) + 3

        # Independent will make us draw 1 sample independently for each batch dim. So sample will be [K,batch,self.z_dim] -> [K, batch]
        if (self.encoder_type == "gaussian"):
            qz_Gx_obs = td.Independent(td.Normal(loc=mu_enc, scale=std_enc), 1)
            p_z = td.Independent(td.Normal(torch.zeros([self.bs, self.z_dim]).to(self.device),
                                           torch.ones([self.bs, self.z_dim]).to(self.device)), 1)
        elif (self.encoder_type == "studentt"):
            qz_Gx_obs = td.Independent(td.StudentT(loc=mu_enc, scale=std_enc, df=df_enc), 1)
            p_z = td.Independent(td.StudentT(loc=torch.zeros([self.bs, self.z_dim]).to(self.device),
                                             scale=torch.ones([self.bs, self.z_dim]).to(self.device), df=df_enc), 1)
        # print(K)
        z_Gx = qz_Gx_obs.rsample([K])

        z_Gx_flat = z_Gx.reshape([K * self.bs, -1])
        out_decoder = self.dec(z_Gx_flat)
        x_mean = out_decoder.reshape([K, self.bs, 1, 28, 28])

        # BCE
        lpx_Gz = td.Bernoulli(probs=x_mean).log_prob(x.repeat(K, 1, 1, 1, 1))

        lpx_Gz_obs = lpx_Gz.sum([-1, -2, -3])
        logpz = p_z.log_prob(z_Gx)
        logq = qz_Gx_obs.log_prob(z_Gx)

        kl = -torch.mean(torch.logsumexp(logpz - logq,0))
        nll = -torch.mean(torch.logsumexp(lpx_Gz_obs - logq,0))

        # emphasize kl more, by multiplying with 3?.
        loss = -torch.mean(torch.logsumexp(lpx_Gz_obs + (logpz - logq)*beta, 0))
        return z_Gx, loss, mu_enc, std_enc, x_mean, kl, nll

    def impute(self, x, K, mask, S=5, impute_type="single"):
        batch_size = 1
        assert batch_size == 1, "Impute function only works for batch_size=1 due to memory reasons"
        out_encoder = self.enc(x.to(self.device)).detach().cpu()
        mu_enc = out_encoder[..., :self.z_dim]
        std_enc = torch.nn.Softplus()(out_encoder[..., self.z_dim:(2 * self.z_dim)]) + 0.0001
        df_enc = torch.nn.Softplus()(out_encoder[..., (2 * self.z_dim):(3 * self.z_dim)]) + 3

        if (self.encoder_type == "gaussian"):
            qz_Gx_obs = td.Independent(td.Normal(loc=mu_enc, scale=std_enc), 1)
            p_z = td.Independent(td.Normal(torch.zeros([self.z_dim]),
                                           torch.ones([self.z_dim])), 1)
        elif (self.encoder_type == "studentt"):
            qz_Gx_obs = td.Independent(td.StudentT(loc=mu_enc, scale=std_enc, df=df_enc), 1)
            p_z = td.Independent(td.StudentT(loc=torch.zeros([self.z_dim]),
                                             scale=torch.ones([self.z_dim]), df=df_enc), 1)

        z_Gx = qz_Gx_obs.rsample([K])
        z_Gx_flat = z_Gx.reshape([K * batch_size, -1])

        out_decoder = self.dec(z_Gx_flat.to(self.device)).detach().cpu()
        x_mean = out_decoder.reshape([K, batch_size, 1, 28, 28])

        lpx_Gz = td.Bernoulli(probs=x_mean).log_prob(x.repeat(K, 1, 1, 1, 1))

        lpx_Gz_obs = lpx_Gz.sum([-1, -2, -3]).reshape([K, batch_size])

        logpz = p_z.log_prob(z_Gx)
        logq = qz_Gx_obs.log_prob(z_Gx)

        xgz = td.Independent(td.Bernoulli(probs=x_mean), 1)

        # sample
        xms = xgz.sample().reshape([K, batch_size, -1])

        # IS weights
        imp_weights = torch.nn.functional.softmax(lpx_Gz_obs + logpz - logq, 0)

        # resample from xms and create number of datasets based on IS weights, S
        if impute_type == "multiple":
            imp_weights /= imp_weights.sum()  # assure they are probabilities summing to 1
            choice = np.random.choice(K, size=[S], replace=True, p=np.array(imp_weights.view(-1)))
            xms = xms[choice,]
            x_mean = x_mean.view(K, 1, -1)[choice,]
            x_prob = x_mean.mean(0)
            xm = majority_vote(xms.view(S, -1)).float().unsqueeze(0)  # [1,S,DIM]

        else:
            x_imp = torch.einsum('ki,kij->ij', imp_weights, xms).clamp(0, 1)  # avoid numerical errors
            x_prob = x_imp
            # test=((imp_weights[:,:,None])*xms).sum([0])
            xm_sample = td.Independent(td.Bernoulli(probs=x_imp), 1)
            xm = xm_sample.sample()

        return xm, x_prob

    def get_df(self, x):
        out_encoder = self.enc(x)
        mu_enc = out_encoder[..., :self.z_dim]
        std_enc = torch.nn.Softplus()(out_encoder[..., self.z_dim:(2 * self.z_dim)]) + 0.0001
        df_enc = torch.nn.Softplus()(out_encoder[..., (2 * self.z_dim):(3 * self.z_dim)]) + 3
        return df_enc

    def sample(self, z):
        x_sample = self.dec(z)
        xgz = td.Independent(td.Bernoulli(probs=x_sample), 1).sample()
        return x_sample, xgz

    def forward(self, x, K, beta=1):
        z, loss, mu_enc, std_enc, mu_dec, kl, nll = self.elbo(x, K, beta)
        return z, loss, mu_enc, std_enc, mu_dec, kl, nll

