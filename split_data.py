#!/usr/bin/env python3
import random, os
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('/mnt/c/Users/Administrator/Desktop/data/GEBCOdata')
OUT_DIR  = Path('/mnt/c/Users/Administrator/Desktop/Experiment/data')
SEED = 42

def main():
    all_dirs = list(DATA_DIR.iterdir())
    tiles = sorted([
        d for d in tqdm(all_dirs, desc='Scanning tiles')
        if d.is_dir() and (d / 'GEBCO2026_1m.nc').exists()
        and (d / 'GEBCO2026_15s.nc').exists()
        and (d / 'cleaned_all.txt').exists()
    ])
    print(f"Found {len(tiles)} complete tiles")
    random.seed(SEED)
    random.shuffle(tiles)
    selected = tiles[:1000]
    train, val = selected[:800], selected[800:1000]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, lst in [('bathy_train.txt', train), ('bathy_val.txt', val)]:
        with open(OUT_DIR / name, 'w') as f:
            for p in lst:
                f.write(str(p) + '\n')
    print(f"Train: {len(train)} tiles, Val: {len(val)} tiles")

if __name__ == '__main__':
    main()
