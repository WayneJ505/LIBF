import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..utils import to_pixel_samples


class BathyMixed(Dataset):

    def __init__(self, base_dataset, inp_size=48, scale_min=1, scale_max=4,
                 sample_q=2304, hr_ratio=0.5, hr_res=1200):
        self.dataset = base_dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.hr_ratio = hr_ratio
        self.hr_res = hr_res

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lr_grid, hr_grid, ship_coord, ship_dep = self.dataset[idx]

        s = random.uniform(self.scale_min, self.scale_max)
        w_hr = round(self.inp_size * s)
        max_offset = self.hr_res - w_hr
        x0 = random.randint(0, max_offset)
        y0 = random.randint(0, max_offset)

        hr_crop = np.ascontiguousarray(hr_grid[y0:y0 + w_hr, x0:x0 + w_hr])
        hr_t = torch.from_numpy(hr_crop).unsqueeze(0).unsqueeze(0)
        lr_t = F.interpolate(hr_t, size=(self.inp_size, self.inp_size),
                             mode='bicubic', align_corners=False)
        lr_t = lr_t.squeeze(0)
        grid_coord, grid_gt = to_pixel_samples(hr_t)
        n_pixels = len(grid_coord)

        n_grid_init = int(self.sample_q * self.hr_ratio)
        n_grid_init = min(n_grid_init, n_pixels)

        grid_idx = np.random.choice(n_pixels, n_grid_init, replace=False)
        sel_coord = [grid_coord[grid_idx]]
        sel_gt = [grid_gt[grid_idx]]
        n_collected = n_grid_init

        n_ship_want = self.sample_q - n_collected
        if n_ship_want > 0:
            hres = self.hr_res
            py = (hres * (ship_coord[:, 0] + 1) - 1) / 2
            px = (hres * (ship_coord[:, 1] + 1) - 1) / 2
            mask = (px >= x0) & (px < x0 + w_hr) & (py >= y0) & (py < y0 + w_hr)
            px_crop = px[mask]
            py_crop = py[mask]
            dep_crop = ship_dep[mask]
            n_ship_avail = len(px_crop)

            if n_ship_avail > 0:
                n_ship = min(n_ship_want, n_ship_avail)
                ship_idx = np.random.choice(n_ship_avail, n_ship, replace=False)
                rel_px = px_crop[ship_idx] - x0
                rel_py = py_crop[ship_idx] - y0

                half = 1.0 / w_hr
                ship_cx = -1 + half + 2 * half * rel_px
                ship_cy = -1 + half + 2 * half * rel_py
                ship_coords = torch.stack([
                    torch.from_numpy(ship_cy),
                    torch.from_numpy(ship_cx)
                ], dim=1).float()
                ship_gts = torch.from_numpy(dep_crop[ship_idx]).float().unsqueeze(-1)

                sel_coord.append(ship_coords)
                sel_gt.append(ship_gts)
                n_collected += n_ship

        shortfall = self.sample_q - n_collected
        if shortfall > 0:
            used = set(grid_idx.tolist())
            remaining = [i for i in range(n_pixels) if i not in used]
            fill_idx = np.random.choice(remaining, shortfall, replace=False)
            sel_coord.append(grid_coord[fill_idx])
            sel_gt.append(grid_gt[fill_idx])

        coord = torch.cat(sel_coord, dim=0)
        gt = torch.cat(sel_gt, dim=0)

        cell = torch.ones(self.sample_q, 2)
        cell[:, 0] *= 2.0 / w_hr
        cell[:, 1] *= 2.0 / w_hr

        return {'inp': lr_t, 'coord': coord, 'cell': cell, 'gt': gt}
