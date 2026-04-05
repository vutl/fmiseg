import argparse
from net.creratemodel import CreateModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl  
from utils.dataset import SegData
import utils.config as config
import os

import platform
import glob
import time
import re

# don't hardcode CUDA_VISIBLE_DEVICES; we'll set it after reading config


def parse_metric_from_checkpoint_name(path, monitor_name):
    filename = os.path.basename(path)
    pattern = re.escape(monitor_name) + r'=([0-9]*\.?[0-9]+)'
    match = re.search(pattern, filename)
    return float(match.group(1)) if match else None


def resolve_best_checkpoint_from_dir(model_dir, monitor_name='val_MIoU', mode='max'):
    if not model_dir or not os.path.isdir(model_dir):
        return None

    info_path = os.path.join(model_dir, 'best_checkpoint.txt')
    if os.path.isfile(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('best_model_path='):
                    candidate = line.split('=', 1)[1].strip()
                    if candidate and os.path.isfile(candidate):
                        return candidate

    scored = []
    fallback = []
    for path in glob.glob(os.path.join(model_dir, '*.ckpt')):
        name = os.path.basename(path)
        if name.startswith('last'):
            continue
        metric_value = parse_metric_from_checkpoint_name(path, monitor_name)
        if metric_value is None:
            fallback.append(path)
        else:
            scored.append((metric_value, path))

    if scored:
        reverse = mode == 'max'
        scored.sort(key=lambda item: item[0], reverse=reverse)
        return scored[0][1]
    if fallback:
        return max(fallback, key=os.path.getmtime)

    last_path = os.path.join(model_dir, 'last.ckpt')
    if os.path.isfile(last_path):
        return last_path
    return None

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/train.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

if __name__ == '__main__':
    args = get_parser()
    dataset_name = getattr(args, 'dataset_name', 'cov19')
    wavelet_type = getattr(args, 'wavelet_type', 'haar')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(getattr(args, 'matmul_precision', 'medium'))
    # load model
    model = CreateModel(args)

    checkpoint_path = getattr(args, 'checkpoint_path', None)
    if checkpoint_path:
        ckpt_path = checkpoint_path
    else:
        model_dir = getattr(args, 'model_save_path', './save_model')
        ckpt_path = resolve_best_checkpoint_from_dir(
            model_dir,
            monitor_name=getattr(args, 'checkpoint_monitor', 'val_MIoU'),
            mode=getattr(args, 'checkpoint_mode', 'max'),
        )
        if not ckpt_path:
            raise FileNotFoundError(f"No checkpoint (.ckpt) found in {model_dir}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    # set device visibility and lightning accelerator/devices
    # args.device may be 'cpu' or an int (gpu id) or a string number
    device_cfg = getattr(args, 'device', None)
    if device_cfg is None:
        # default to first GPU if available
        accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
        devices = 1 if accelerator == 'cuda' else 'auto'
    else:
        # normalize
        try:
            dev_int = int(device_cfg)
        except Exception:
            dev_int = None
        if device_cfg == 'cpu' or dev_int is None and not torch.cuda.is_available():
            accelerator = 'cpu'
            devices = 'auto'
        else:
            accelerator = 'cuda'
            # if an integer GPU id provided, set CUDA_VISIBLE_DEVICES and use that card
            if dev_int is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_int)
                devices = 1
            else:
                devices = 1

    # dataloader (adjust workers on Windows)
    if platform.system().lower().startswith('win'):
        test_workers = 0
    else:
        # try to use a few workers
        test_workers = min(8, max(0, args.valid_batch_size))

    ds_test = SegData(dataname=dataset_name,
                    csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test',
                    wavelet_type=wavelet_type,
                    auto_prompt_from_mask=getattr(args, 'auto_prompt_from_mask', False))
    dl_test = DataLoader(
        ds_test,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=test_workers,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        limit_test_batches=getattr(args, 'limit_test_batches', 1.0),
    )
    model.eval()
    trainer.test(model, dl_test)
