import numpy as np
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """CDDPM utility aligned with Eqs. (9)-(14) and Algorithms 1-2."""

    def __init__(self, timesteps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

    def to_device(self, device):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        out = arr.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(y0)
        return self._extract(self.sqrt_alphas_cumprod, t, y0.shape) * y0 + self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, y0.shape
        ) * noise

    def q_posterior(self, y0: torch.Tensor, yt: torch.Tensor, t: torch.Tensor):
        mean = self._extract(self.posterior_mean_coef1, t, yt.shape) * y0 + self._extract(
            self.posterior_mean_coef2, t, yt.shape
        ) * yt
        var = self._extract(self.posterior_variance, t, yt.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, yt.shape)
        return mean, var, log_var

    def calculate_loss(self, model, x_start, t, condition_feature, raw_bcg, lambda_weight: float = 1.0):
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        pred_noise = model(torch.cat([x_t, raw_bcg], dim=1), t, condition_feature)

        l_condition = F.mse_loss(pred_noise, noise)

        pred_y0 = (x_t - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * pred_noise) / self._extract(
            self.sqrt_alphas_cumprod, t, x_t.shape
        )
        true_mean, _, true_log_var = self.q_posterior(x_start, x_t, t)
        pred_mean, _, pred_log_var = self.q_posterior(pred_y0, x_t, t)
        l_kl = 0.5 * (
            pred_log_var
            - true_log_var
            + (torch.exp(true_log_var) + (true_mean - pred_mean) ** 2) / torch.exp(pred_log_var)
            - 1.0
        )
        l_kl_condition = l_kl.mean()
        l_mix = l_kl_condition + lambda_weight * l_condition
        return l_mix, {"Lmix": l_mix.detach(), "Lcondition": l_condition.detach(), "LKL_condition": l_kl_condition.detach()}

    @torch.no_grad()
    def p_sample(self, model, x, t, condition_feature, raw_bcg):
        pred_noise = model(torch.cat([x, raw_bcg], dim=1), t, condition_feature)
        alpha_t = self._extract(self.alphas, t, x.shape)
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
        beta_t = self._extract(self.betas, t, x.shape)
        sigma_t = torch.sqrt(self._extract(self.posterior_variance, t, x.shape))

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x)
        return mean + nonzero_mask * sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition_feature, raw_bcg, device=None, init_mask=None):
        if device is None:
            device = raw_bcg.device
        if init_mask is None:
            x = torch.randn(shape, device=device)
        else:
            x = init_mask.to(device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition_feature, raw_bcg)
        return x
