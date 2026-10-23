import os
import argparse
import warnings

import numpy as np
import torch
import xarray as xr
from pathlib import Path

from .utils import (
    make_coord, compute_psnr, compute_rmse,
    GLOBAL_DEPTH_MIN, GLOBAL_DEPTH_MAX,
    batched_predict, load_model,
)

_COORD_VARS = {'lat', 'latitude', 'lon', 'longitude', 'x', 'y'}


def _detect_var(ds):
    """Auto-detect the main 2D data variable in a NetCDF dataset."""
    candidates = []
    for name, da in ds.data_vars.items():
        if name.lower() in _COORD_VARS:
            continue
        if da.ndim >= 2:
            candidates.append((name, da.size))
    if not candidates:
        raise ValueError("No 2D data variable found in NetCDF file. "
                         "Use --var to specify the variable name.")
    candidates.sort(key=lambda x: -x[1])
    name, _ = candidates[0]
    return name


def _normalize(data, d_min, d_max):
    return (data - d_min) / (d_max - d_min) * 2 - 1


def _denormalize(data_norm, d_min, d_max):
    return (data_norm + 1) / 2 * (d_max - d_min) + d_min


def super_resolve(model, lr_tensor, out_h, out_w, device):
    coord = make_coord((out_h, out_w), flatten=True).to(device).unsqueeze(0)
    cell = torch.ones(1, out_h * out_w, 2).to(device)
    cell[:, :, 0] *= 2.0 / out_h
    cell[:, :, 1] *= 2.0 / out_w
    pred = batched_predict(model, lr_tensor, coord, cell)
    return pred.view(out_h, out_w).cpu()


def main():
    parser = argparse.ArgumentParser(
        description='Bathymetry super-resolution with LIIF')
    parser.add_argument('--input', required=True,
                        help='Path to input NetCDF file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--scale', type=int, required=True,
                        help='Super-resolution factor (e.g., 2 or 4)')
    parser.add_argument('--output', default=None,
                        help='Output NetCDF path (default: <input>_x<scale>.nc)')
    parser.add_argument('--var', default=None,
                        help='Variable name in NetCDF (auto-detect if omitted)')
    parser.add_argument('--depth_min', type=float, default=GLOBAL_DEPTH_MIN,
                        help=f'Minimum depth value (default: {GLOBAL_DEPTH_MIN})')
    parser.add_argument('--depth_max', type=float, default=GLOBAL_DEPTH_MAX,
                        help=f'Maximum depth value (default: {GLOBAL_DEPTH_MAX})')
    parser.add_argument('--gt', default=None,
                        help='Optional ground truth NC for PSNR/RMSE comparison')
    parser.add_argument('--gt_var', default=None,
                        help='Variable name in ground truth NC')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Force CPU even if CUDA is available')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    d_min = args.depth_min
    d_max = args.depth_max

    print(f"Device: {device}")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds = xr.open_dataset(args.input)
    var_name = args.var or _detect_var(ds)
    data = ds[var_name].values.astype(np.float32)
    ds.close()
    h_in, w_in = data.shape
    print(f"Input:  {args.input}  shape=({h_in}, {w_in})  var='{var_name}'")

    data_norm = _normalize(data, d_min, d_max)
    lr = torch.from_numpy(data_norm).unsqueeze(0).unsqueeze(0).to(device)

    h_out, w_out = h_in * args.scale, w_in * args.scale
    print(f"Output: {h_out}x{w_out} (scale={args.scale}x)")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded")

    sr_norm = super_resolve(model, lr, h_out, w_out, device)
    sr_depth = _denormalize(sr_norm.numpy(), d_min, d_max)

    out_path = args.output or f"{Path(args.input).stem}_x{args.scale}.nc"
    out_ds = xr.Dataset(
        {var_name: (['lat', 'lon'], sr_depth)},
        coords={'lat': np.arange(h_out), 'lon': np.arange(w_out)},
    )
    out_ds.to_netcdf(out_path)
    print(f"Saved: {out_path}  range=[{sr_depth.min():.1f}, {sr_depth.max():.1f}]")

    if args.gt:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gt_ds = xr.open_dataset(args.gt)
        gt_var = args.gt_var or _detect_var(gt_ds)
        gt = gt_ds[gt_var].values.astype(np.float32)
        gt_ds.close()
        gt_norm = _normalize(gt, d_min, d_max)

        if gt.shape != sr_norm.shape:
            print(f"Warning: GT shape {gt.shape} != SR shape {sr_norm.shape}, "
                  f"cropping to smaller size")
            h_min = min(gt.shape[0], sr_norm.shape[0])
            w_min = min(gt.shape[1], sr_norm.shape[1])
            gt_n = gt_norm[:h_min, :w_min]
            sr_n = sr_norm[:h_min, :w_min]
            gt_d = gt[:h_min, :w_min]
        else:
            gt_n = gt_norm
            sr_n = sr_norm.numpy()
            gt_d = gt

        psnr = compute_psnr(torch.from_numpy(sr_n), torch.from_numpy(gt_n))
        rmse = compute_rmse(torch.from_numpy(sr_n), torch.from_numpy(gt_n),
                            torch.tensor(d_min), torch.tensor(d_max))
        print(f"PSNR: {psnr:.2f} dB")
        print(f"RMSE: {rmse:.2f} m")


if __name__ == '__main__':
    main()
