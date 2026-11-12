"""Versioned local NSS descriptors; no learned parameters or external CPU fits."""
import torch
from torch import nn
from torch.nn import functional as F

MODES = {'legacy': (0, 6), 'mscn_stats6_v1': (1, 6), 'mscn_ggd_v1': (2, 2)}


def validate_nss(config):
    mode = config.get('descriptor', 'legacy')
    if mode not in MODES:
        raise ValueError(f'Unknown NSS descriptor: {mode}')
    if int(config.get('nss_dim', MODES[mode][1])) != MODES[mode][1]:
        raise ValueError(f'{mode} requires nss_dim={MODES[mode][1]}')
    return mode, MODES[mode][1]


class StandardMSCNDescriptor(nn.Module):
    """Gaussian 7x7 sigma=7/6, replicate border, 8-bit stabilizer C=1.

    RGB input is [0,1]. Zero-mean GGD uses raw second moment as variance,
    rather than subtracting the finite-sample patch mean.
    """
    def __init__(self, mode):
        super().__init__()
        if mode not in ('mscn_stats6_v1', 'mscn_ggd_v1'):
            raise ValueError(mode)
        self.mode = mode
        self.output_dim = MODES[mode][1]
        x = torch.arange(-3, 4, dtype=torch.float64)
        g = torch.exp(-x.square() / (2 * (7 / 6) ** 2)); g /= g.sum()
        self.register_buffer('kernel', (g[:, None]*g[None, :])[None, None].float(), persistent=False)
        alpha = torch.linspace(.2, 10, 9801, dtype=torch.float64)
        ratio = torch.exp(2*torch.lgamma(2/alpha)-torch.lgamma(1/alpha)-torch.lgamma(3/alpha))
        self.register_buffer('alpha_grid', alpha.float(), persistent=False)
        self.register_buffer('ratio_grid', ratio.float(), persistent=False)

    def mscn(self, rgb):
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            gray = (rgb.float()*rgb.new_tensor([.299,.587,.114],dtype=torch.float32)[None,:,None,None]).sum(1,keepdim=True)
            blur = lambda x: F.conv2d(F.pad(x,(3,3,3,3),mode='replicate'),self.kernel.float())
            mu = blur(gray)
            variance = (blur(gray.square())-mu.square()).clamp_min(0)
            z = (gray-mu)/(torch.where(variance>1e-12,variance.clamp_min(1e-12).sqrt(),torch.zeros_like(variance))+1/255)
        return gray,z

    def fit_ggd_moments(self, absolute, second):
        # E|X|^2/E[X^2] = Gamma(2/a)^2 / (Gamma(1/a) Gamma(3/a)).
        ratio = absolute.square()/second.clamp_min(1e-12)
        r = ratio.clamp(self.ratio_grid[0],self.ratio_grid[-1])
        hi = torch.searchsorted(self.ratio_grid,r.contiguous()).clamp(1,len(self.ratio_grid)-1)
        lo = hi-1
        fraction = (r-self.ratio_grid[lo])/(self.ratio_grid[hi]-self.ratio_grid[lo])
        alpha = self.alpha_grid[lo]+fraction*(self.alpha_grid[hi]-self.alpha_grid[lo])
        # Degenerate point masses have no identifiable GGD shape.
        alpha = torch.where(second>1e-10,alpha,torch.full_like(alpha,2))
        return alpha,second

    def forward(self, rgb, out_hw):
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            gray,z = self.mscn(rgb)
            pool = lambda x: F.adaptive_avg_pool2d(x,out_hw)
            mean,second,absolute = pool(z),pool(z.square()),pool(z.abs())
            if self.mode == 'mscn_ggd_v1':
                features = self.fit_ggd_moments(absolute,second)
            else:
                variance = (second-mean.square()).clamp_min(0)
                m3 = pool(z.pow(3)); m4 = pool(z.pow(4))
                third = m3-3*mean*second+2*mean.pow(3)
                fourth = m4-4*mean*m3+6*mean.square()*second-3*mean.pow(4)
                std = variance.clamp_min(1e-12).sqrt()
                skew = torch.where(variance>1e-10,third/std.pow(3),torch.zeros_like(mean))
                kurt = torch.where(variance>1e-10,fourth/std.pow(4)-3,torch.zeros_like(mean))
                std = torch.where(variance>1e-10,std,torch.zeros_like(std))
                dx = F.pad(gray[:,:,:,1:]-gray[:,:,:,:-1],(0,1,0,0))
                dy = F.pad(gray[:,:,1:,:]-gray[:,:,:-1,:],(0,0,0,1))
                gradient = (dx.square()+dy.square()+1e-12).sqrt()-1e-6
                features = (mean,std,absolute,skew,kurt,pool(gradient))
            return torch.cat(features,1).flatten(2).transpose(1,2)
