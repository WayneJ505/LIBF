import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import GLOBAL_DEPTH_MIN, GLOBAL_DEPTH_MAX


class BathyTileDataset(Dataset):

    def __init__(self, index_file, repeat=1, hr_res=1200):
        with open(index_file) as f:
            self.tile_dirs = [Path(line.strip()) for line in f if line.strip()]
        self.repeat = repeat
        self.rng = GLOBAL_DEPTH_MAX - GLOBAL_DEPTH_MIN
        self.hr_res = hr_res

    def __len__(self):
        return len(self.tile_dirs) * self.repeat

    def __getitem__(self, idx):
        d = self.tile_dirs[idx % len(self.tile_dirs)]

        lr = xr.open_dataset(str(d / 'GEBCO2026_1m.nc'))
        z = lr['z'].values.astype(np.float32)
        lr.close()
        lr_norm = (z - GLOBAL_DEPTH_MIN) / self.rng * 2 - 1

        hr = xr.open_dataset(str(d / 'GEBCO2026_15s.nc'))
        elev = hr['elevation'].values.astype(np.float32)
        hr.close()
        hr_norm = (elev - GLOBAL_DEPTH_MIN) / self.rng * 2 - 1

        pts = pd.read_csv(str(d / 'cleaned_all.txt'), sep='\t')
        lon = pts['longitude'].values.astype(np.float32)
        lat = pts['latitude'].values.astype(np.float32)
        dep = pts['depth'].values.astype(np.float32)

        name = d.name
        lat_start = float(name.split('_')[0][3:])
        lon_start = float(name.split('_')[1][3:])

        pix_x = (lon - lon_start) / 5.0 * self.hr_res
        pix_y = (lat - lat_start) / 5.0 * self.hr_res
        half = 1.0 / self.hr_res
        coord_x = -1 + half + 2 * half * pix_x
        coord_y = -1 + half + 2 * half * pix_y
        ship_coord = np.stack([coord_y, coord_x], axis=1).astype(np.float32)

        dep_norm = (dep - GLOBAL_DEPTH_MIN) / self.rng * 2 - 1

        return lr_norm, hr_norm, ship_coord, dep_norm.astype(np.float32)
