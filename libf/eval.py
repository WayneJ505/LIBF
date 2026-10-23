import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .utils import (
    make_coord, compute_psnr, compute_rmse,
    GLOBAL_DEPTH_MIN, GLOBAL_DEPTH_MAX,
    load_model,
)

RNG = GLOBAL_DEPTH_MAX - GLOBAL_DEPTH_MIN
SCALES = [2, 4, 6, 8]
EVAL_BSIZE = 60000


@torch.no_grad()
def eval_grid(model, device, tiles, n_channels, cache_path=None):
    results = {s: {'psnr': [], 'rmse': []} for s in SCALES}

    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        done_tiles = set()
        for s_k in cached:
            results[int(s_k)] = cached[s_k]
        if results:
            done_tiles = set(results[SCALES[0]].get('tiles_done', []))
            print(f"  [resume] {len(done_tiles)} tiles cached, {len(tiles)-len(done_tiles)} remaining")
    else:
        done_tiles = set()

    for i, td in enumerate(tqdm(tiles, desc='Grid')):
        tp = Path(td)
        if tp.name in done_tiles:
            continue
        hr_ds = xr.open_dataset(str(tp / 'GEBCO2026_15s.nc'))
        hr = hr_ds['elevation'].values.astype('float32'); hr_ds.close()
        hr_11 = (hr - GLOBAL_DEPTH_MIN) / RNG * 2 - 1

        for s in SCALES:
            hc = (1200 // s) * s; wc = (1200 // s) * s
            y0 = (1200 - hc) // 2; x0 = (1200 - wc) // 2
            hr_crop = hr_11[y0:y0 + hc, x0:x0 + wc]
            lh, lw = hc // s, wc // s
            gt = torch.from_numpy(hr_crop)

            lr = F.interpolate(gt.unsqueeze(0).unsqueeze(0),
                               size=(lh, lw), mode='bicubic', align_corners=False)

            inp = lr.squeeze(0).unsqueeze(0).to(device)
            if n_channels == 3:
                inp = inp.repeat(1, 3, 1, 1)
            coord = make_coord((hc, wc), flatten=True).to(device).unsqueeze(0)
            cell = torch.ones(1, hc * wc, 2).to(device)
            cell[:, :, 0] *= 2.0 / hc; cell[:, :, 1] *= 2.0 / wc
            model.gen_feat(inp)
            preds = []
            for ql in range(0, hc * wc, EVAL_BSIZE):
                qr = min(ql + EVAL_BSIZE, hc * wc)
                preds.append(model.query_rgb(coord[:, ql:qr], cell[:, ql:qr]).cpu())
            pred_all = torch.cat(preds, dim=1).view(1, hc, wc, -1)
            pred = pred_all[0, :, :, 0] if pred_all.shape[-1] >= 1 else pred_all[0]

            psnr = compute_psnr(pred, gt)
            rmse = compute_rmse(pred, gt, torch.tensor(GLOBAL_DEPTH_MIN), torch.tensor(GLOBAL_DEPTH_MAX))
            results[s]['psnr'].append(psnr)
            results[s]['rmse'].append(rmse)

        for s in SCALES:
            results[s].setdefault('tiles_done', []).append(tp.name)

        if cache_path and (i + 1) % 5 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({str(k): v for k, v in results.items()}, f)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f)
    return results


@torch.no_grad()
def eval_bicubic(tiles, cache_path=None):
    results = {s: {'psnr': [], 'rmse': []} for s in SCALES}

    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        done_tiles = set()
        for s_k in cached:
            results[int(s_k)] = cached[s_k]
        if results:
            done_tiles = set(results[SCALES[0]].get('tiles_done', []))
            print(f"  [resume] {len(done_tiles)} tiles cached, {len(tiles)-len(done_tiles)} remaining")
    else:
        done_tiles = set()

    for i, td in enumerate(tqdm(tiles, desc='Bicubic')):
        tp = Path(td)
        if tp.name in done_tiles:
            continue
        hr_ds = xr.open_dataset(str(tp / 'GEBCO2026_15s.nc'))
        hr = hr_ds['elevation'].values.astype('float32'); hr_ds.close()
        hr_11 = (hr - GLOBAL_DEPTH_MIN) / RNG * 2 - 1

        for s in SCALES:
            hc = (1200 // s) * s; wc = (1200 // s) * s
            y0 = (1200 - hc) // 2; x0 = (1200 - wc) // 2
            hr_crop = hr_11[y0:y0 + hc, x0:x0 + wc]
            lh, lw = hc // s, wc // s
            gt = torch.from_numpy(hr_crop)

            lr = F.interpolate(gt.unsqueeze(0).unsqueeze(0),
                               size=(lh, lw), mode='bicubic', align_corners=False)
            pred = F.interpolate(lr, size=(hc, wc),
                                 mode='bicubic', align_corners=False).squeeze().squeeze()

            psnr = compute_psnr(pred, gt)
            rmse = compute_rmse(pred, gt, torch.tensor(GLOBAL_DEPTH_MIN), torch.tensor(GLOBAL_DEPTH_MAX))
            results[s]['psnr'].append(psnr)
            results[s]['rmse'].append(rmse)

        for s in SCALES:
            results[s].setdefault('tiles_done', []).append(tp.name)

        if cache_path and (i + 1) % 10 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({str(k): v for k, v in results.items()}, f)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f)
    return results


@torch.no_grad()
def eval_ship(model, device, tiles, n_channels):
    ship_r, ship_m, ship_r2 = [], [], []

    for td in tqdm(tiles, desc='Ship'):
        tp = Path(td); name = tp.name
        la0 = float(name.split('_')[0][3:]); lo0 = float(name.split('_')[1][3:])

        hr_ds = xr.open_dataset(str(tp / 'GEBCO2026_15s.nc'))
        hr = hr_ds['elevation'].values.astype('float32'); hr_ds.close()
        hr_11 = (hr - GLOBAL_DEPTH_MIN) / RNG * 2 - 1

        pts = pd.read_csv(str(tp / 'cleaned_all.txt'), sep='\t')
        lon = pts['longitude'].values; lat = pts['latitude'].values; dep = pts['depth'].values
        half = 1.0 / 1200
        px = (lon - lo0) / 5.0 * 1200; py = (lat - la0) / 5.0 * 1200
        v = (px >= 0) & (px <= 1200) & (py >= 0) & (py <= 1200)
        if v.sum() == 0: continue
        px, py, dep = px[v], py[v], dep[v]

        cx = -1 + half + 2 * half * px
        cy = -1 + half + 2 * half * py
        ship_coord = torch.from_numpy(np.stack([cy, cx], axis=1)).float().to(device).unsqueeze(0)

        inp = torch.from_numpy(hr_11).unsqueeze(0).unsqueeze(0).to(device)
        if n_channels == 3:
            inp = inp.repeat(1, 3, 1, 1)

        n = len(dep)
        cell = torch.ones(1, n, 2).to(device)
        cell[:, :, 0] *= 2.0 / hr_11.shape[0]; cell[:, :, 1] *= 2.0 / hr_11.shape[1]

        model.gen_feat(inp)
        preds = []
        for ql in range(0, n, EVAL_BSIZE):
            qr = min(ql + EVAL_BSIZE, n)
            preds.append(model.query_rgb(ship_coord[:, ql:qr], cell[:, ql:qr]).cpu())
        pred = torch.cat(preds, dim=1).squeeze(0)
        pred_n = pred[:, 0].numpy() if pred.dim() == 2 and pred.shape[1] > 1 else pred.squeeze(-1).numpy()

        pred_depth = (pred_n + 1) / 2 * RNG + GLOBAL_DEPTH_MIN
        err = pred_depth - dep
        ship_r.append(np.sqrt(np.mean(err ** 2)))
        ship_m.append(np.mean(np.abs(err)))
        ss = np.sum(err ** 2); st = np.sum((dep - dep.mean()) ** 2)
        ship_r2.append(1 - ss / st if st > 0 else 0)

    return ship_r, ship_m, ship_r2


def eval_ship_bilinear(tiles):
    ship_r, ship_m, ship_r2 = [], [], []
    for td in tqdm(tiles, desc='Ship(BL)'):
        tp = Path(td); name = tp.name
        la0 = float(name.split('_')[0][3:]); lo0 = float(name.split('_')[1][3:])
        hr = xr.open_dataset(str(tp / 'GEBCO2026_15s.nc'))['elevation'].values.astype('float32')
        pts = pd.read_csv(str(tp / 'cleaned_all.txt'), sep='\t')
        lon = pts['longitude'].values; lat = pts['latitude'].values; dep = pts['depth'].values
        px = (lon - lo0) / 5.0 * 1200; py = (lat - la0) / 5.0 * 1200
        v = (px >= 0) & (px <= 1200) & (py >= 0) & (py <= 1200)
        if v.sum() == 0: continue
        px, py, dep = px[v], py[v], dep[v]
        hr_t = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)
        px_n = (torch.from_numpy(px).float() / 1200) * 2 - 1
        py_n = (torch.from_numpy(py).float() / 1200) * 2 - 1
        grid = torch.stack([px_n, py_n], dim=1).unsqueeze(0).unsqueeze(0)
        pred_depth = F.grid_sample(hr_t, grid, mode='bilinear',
                                   align_corners=True, padding_mode='border')
        pred_depth = pred_depth.squeeze().numpy()
        err = pred_depth - dep
        ship_r.append(np.sqrt(np.mean(err ** 2)))
        ship_m.append(np.mean(np.abs(err)))
        ss = np.sum(err ** 2); st = np.sum((dep - dep.mean()) ** 2)
        ship_r2.append(1 - ss / st if st > 0 else 0)
    return ship_r, ship_m, ship_r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_index', default='index/bathy_val.txt')
    parser.add_argument('--max_tiles', type=int, default=20)
    parser.add_argument('--cache_dir', default='output/eval_cache')
    parser.add_argument('--no_cache', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.val_index) as f:
        tiles = [l.strip() for l in f if l.strip()]
    tiles = tiles[:args.max_tiles]
    print(f"Tiles: {len(tiles)}")

    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache

    all_grid = {}
    all_ship = {}

    models_cfg = [
        ('output/bathy-ship/epoch-best.pth', 'Pure Ship'),
        ('output/bathy-grid/epoch-best.pth', 'Pure Grid'),
    ]

    cached_models = []

    print("\n" + "=" * 80)
    print("Bicubic baseline")
    print("=" * 80)
    cache_path = cache_dir / 'bicubic.json' if use_cache else None
    all_grid['Bicubic'] = eval_bicubic(tiles, cache_path=cache_path)
    all_ship['Bicubic'] = eval_ship_bilinear(tiles)

    for ckpt_path, label in models_cfg:
        if not os.path.exists(ckpt_path):
            print(f"\nSkip {label}: checkpoint not found at {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', '?')
        print(f"\n{'=' * 80}")
        print(f"Evaluating {label} (epoch {epoch})")
        print(f"{'=' * 80}")

        model = load_model(ckpt_path, device)
        model.eval()

        cache_key = label.lower().replace(' ', '_')
        cache_path = cache_dir / f'{cache_key}.json' if use_cache else None
        all_grid[label] = eval_grid(model, device, tiles, 1, cache_path=cache_path)
        all_ship[label] = eval_ship(model, device, tiles, 1)

        del model
        torch.cuda.empty_cache()

    labels = [lb for lb in ['Bicubic', 'Pure Ship', 'Pure Grid'] if lb in all_grid]

    os.makedirs('output', exist_ok=True)
    lines = []

    lines.append(f"\n{'='*100}")
    lines.append(f"Grid SR PSNR (dB) / RMSE (m) — {len(tiles)} val tiles")
    header = f"  {'Scale':>5s}"
    for lb in labels:
        header += f" | {lb:>20s}"
    lines.append(header)
    lines.append('-' * len(header))
    for s in SCALES:
        row = f"  x{s:<3d} "
        for lb in labels:
            if s in all_grid[lb]:
                p = np.mean(all_grid[lb][s]['psnr'])
                r = np.mean(all_grid[lb][s]['rmse'])
                row += f" | {p:>6.2f} dB {r:>7.2f}m"
            else:
                row += f" | {'-':>15s}"
        lines.append(row)

    lines.append(f"\n{'='*100}")
    lines.append(f"Ship Point Evaluation (LIIF from 15s grid, query at ship coords)")
    ship_header = f"{'':>12s}"
    for lb in labels:
        ship_header += f" | {lb:>20s}"
    lines.append(ship_header)
    lines.append('-' * len(ship_header))
    for metric, key in [('RMSE (m)', 0), ('MAE (m)', 1), ('R²', 2)]:
        row = f"  {metric:>12s}"
        for lb in labels:
            v = np.mean(all_ship[lb][key])
            fmt = f'{v:>20.2f}' if key < 2 else f'{v:>20.4f}'
            row += f" | {fmt}"
        lines.append(row)

    out_path = 'output/eval_results.txt'
    with open(out_path, 'w') as f:
        for line in lines:
            print(line)
            f.write(line + '\n')
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
