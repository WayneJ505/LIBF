import os
import random
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.bathy_tile import BathyTileDataset
from .datasets.bathy_wrapper import BathyMixed
from .models.models import make as make_model
from .utils import Averager, Timer, compute_psnr, compute_rmse, GLOBAL_DEPTH_MIN, GLOBAL_DEPTH_MAX


def make_data_loader(spec, train=True):
    ds_cfg = spec['dataset']
    wr_cfg = spec['wrapper']
    bs = spec['batch_size']
    nw = spec.get('num_workers', 0)

    if ds_cfg['name'] == 'bathy-tile':
        ds = BathyTileDataset(**ds_cfg['args'])
    else:
        raise ValueError(f"Unknown dataset: {ds_cfg['name']}")

    if wr_cfg['name'] == 'bathy-mixed':
        ds = BathyMixed(ds, **wr_cfg['args'])
    else:
        raise ValueError(f"Unknown wrapper: {wr_cfg['name']}")

    return DataLoader(ds, batch_size=bs, shuffle=train,
                      num_workers=nw, pin_memory=True,
                      drop_last=train)


def train_epoch(loader, model, optimizer, device, data_norm, loss_fn):
    model.train()
    avg = Averager()
    t = Timer()
    t.s()

    inp_sub = data_norm['inp_sub'].view(1, -1, 1, 1).to(device)
    inp_div = data_norm['inp_div'].view(1, -1, 1, 1).to(device)
    gt_sub = data_norm['gt_sub'].view(1, 1, -1).to(device)
    gt_div = data_norm['gt_div'].view(1, 1, -1).to(device)

    for batch in tqdm(loader, desc='Train', leave=False):
        inp = batch['inp'].to(device)
        coord = batch['coord'].to(device)
        cell = batch['cell'].to(device)
        gt = batch['gt'].to(device)

        inp = (inp - inp_sub) / inp_div
        gt = (gt - gt_sub) / gt_div

        pred = model(inp, coord, cell)
        loss = loss_fn(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg.add(loss.item())

    return avg.item(), t.t()


def validate(loader, model, device, data_norm, loss_fn):
    model.eval()
    avg = Averager()
    rmse_vals = []
    psnr_vals = []

    inp_sub = data_norm['inp_sub'].view(1, -1, 1, 1).to(device)
    inp_div = data_norm['inp_div'].view(1, -1, 1, 1).to(device)
    gt_sub = data_norm['gt_sub'].view(1, 1, -1).to(device)
    gt_div = data_norm['gt_div'].view(1, 1, -1).to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            inp = batch['inp'].to(device)
            coord = batch['coord'].to(device)
            cell = batch['cell'].to(device)
            gt = batch['gt'].to(device)

            inp = (inp - inp_sub) / inp_div
            gt = (gt - gt_sub) / gt_div

            pred = model(inp, coord, cell)
            loss = loss_fn(pred, gt)

            avg.add(loss.item())
            pred_01 = pred * gt_div + gt_sub
            gt_01 = gt * gt_div + gt_sub
            rmse_vals.append(compute_rmse(pred_01.squeeze(-1), gt_01.squeeze(-1),
                                          torch.tensor(GLOBAL_DEPTH_MIN),
                                          torch.tensor(GLOBAL_DEPTH_MAX)))
            psnr_vals.append(compute_psnr(pred_01.squeeze(-1), gt_01.squeeze(-1)))

    return avg.item(), sum(rmse_vals) / len(rmse_vals), sum(psnr_vals) / len(psnr_vals)


def main(config_path, resume_path=None, hr_ratio=None, out_name=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if hr_ratio is not None:
        config['train_dataset']['wrapper']['args']['hr_ratio'] = hr_ratio

    config_name = out_name if out_name else os.path.splitext(os.path.basename(config_path))[0]
    if hr_ratio is not None and out_name is None:
        config_name = f"{config_name}-hr{hr_ratio}"
    hr = config['train_dataset']['wrapper']['args'].get('hr_ratio', 0.5)

    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | hr_ratio={hr}")

    train_loader = make_data_loader(config['train_dataset'], train=True)
    val_loader = make_data_loader(config['val_dataset'], train=False)

    model = make_model(config['model']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    opt_cfg = config['optimizer']
    if opt_cfg['name'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **opt_cfg['args'])
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")

    loss_name = config.get('loss', 'l1')
    if loss_name == 'huber':
        loss_delta = config.get('huber_delta', 0.1)
        loss_fn = nn.HuberLoss(delta=loss_delta)
        print(f"Loss: Huber (delta={loss_delta})")
    else:
        loss_fn = nn.L1Loss()
        print(f"Loss: L1")

    epoch_max = config['epoch_max']
    epoch_val = config['epoch_val']
    epoch_save = config['epoch_save']

    patience_lr = config.get('patience_lr', 8)
    patience_es = config.get('patience_es', 15)

    mslr = config.get('multi_step_lr', None)
    if mslr is not None:
        scheduler = MultiStepLR(optimizer, milestones=mslr['milestones'],
                                gamma=mslr['gamma'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=patience_lr, min_lr=1e-7)

    data_norm_cfg = config.get('data_norm', {
        'inp': {'sub': [0], 'div': [1]},
        'gt': {'sub': [0], 'div': [1]}
    })
    data_norm = {
        'inp_sub': torch.FloatTensor(data_norm_cfg['inp']['sub']),
        'inp_div': torch.FloatTensor(data_norm_cfg['inp']['div']),
        'gt_sub': torch.FloatTensor(data_norm_cfg['gt']['sub']),
        'gt_div': torch.FloatTensor(data_norm_cfg['gt']['div']),
    }

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'output', config_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    timer_total = Timer()
    timer_total.s()

    start_epoch = 1
    best_val_psnr = -float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, epoch_max + 1):
        t = Timer()
        t.s()
        train_loss, train_time = train_epoch(
            train_loader, model, optimizer, device, data_norm, loss_fn)

        log = (f"Epoch {epoch:>4d}/{epoch_max} | "
               f"train_loss={train_loss:.6f} | "
               f"lr={optimizer.param_groups[0]['lr']:.2e} | "
               f"time={train_time:.0f}s")

        if epoch % epoch_val == 0:
            val_loss, val_rmse, val_psnr = validate(val_loader, model, device, data_norm, loss_fn)
            log += f" | val_loss={val_loss:.6f} | val_rmse={val_rmse:.1f}m | val_PSNR={val_psnr:.2f}dB"

            scheduler.step(val_loss)

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                epochs_no_improve = 0
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_val_psnr': best_val_psnr},
                           os.path.join(save_dir, 'epoch-best.pth'))
                log += " *"
            else:
                epochs_no_improve += epoch_val
                log += f" | no_impr={epochs_no_improve}"

            if epochs_no_improve >= patience_es:
                print(log, flush=True)
                print(f"Early stopping at epoch {epoch}")
                break

        print(log, flush=True)

        if epoch % epoch_save == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_psnr': best_val_psnr},
                       os.path.join(save_dir, f'epoch-{epoch}.pth'))

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_psnr': best_val_psnr},
                   os.path.join(save_dir, 'epoch-last.pth'))

    print(f"\nTotal: {timer_total.t():.0f}s | Best val_PSNR: {best_val_psnr:.2f}dB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train-bathy.yaml')
    parser.add_argument('--hr_ratio', type=float, default=None,
                        help='Override hr_ratio (0=pure ship, 1=pure grid)')
    parser.add_argument('--name', default=None, help='Output subfolder name')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    main(args.config, args.resume, args.hr_ratio, args.name)
