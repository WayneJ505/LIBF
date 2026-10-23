import time

import torch
import numpy as np

GLOBAL_DEPTH_MIN = -12000.0
GLOBAL_DEPTH_MAX = 0.0


class Averager:

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer:

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    coord = make_coord(img.shape[-2:])
    val = img.view(img.shape[0], -1).permute(1, 0)
    return coord, val


def denormalize_depth(depth_norm, d_min, d_max):
    depth_norm = torch.as_tensor(depth_norm, dtype=torch.float32)
    return (depth_norm + 1) / 2 * (d_max - d_min) + d_min


def compute_psnr(pred, gt, max_val=2.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def compute_rmse(pred_norm, gt_norm, d_min, d_max):
    pred = denormalize_depth(pred_norm, d_min, d_max)
    gt = denormalize_depth(gt_norm, d_min, d_max)
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


@torch.no_grad()
def batched_predict(model, inp, coord, cell, bsize=30000):
    n = coord.shape[1]
    model.gen_feat(inp)
    preds = []
    for ql in range(0, n, bsize):
        qr = min(ql + bsize, n)
        preds.append(model.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
    return torch.cat(preds, dim=1)


def load_model(checkpoint, device):
    from .models.models import make as make_model

    ckpt = torch.load(checkpoint, map_location=device)
    model_spec = {
        'name': 'liif',
        'args': {
            'encoder_spec': {
                'name': 'rdn',
                'args': {'no_upsampling': True, 'n_colors': 1},
            },
            'imnet_spec': {
                'name': 'mlp',
                'args': {'out_dim': 1, 'hidden_list': [256, 256, 256, 256]},
            },
            'local_ensemble': True,
            'feat_unfold': True,
            'cell_decode': True,
        },
    }
    model = make_model(model_spec).to(device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model
