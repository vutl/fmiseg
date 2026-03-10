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

# don't hardcode CUDA_VISIBLE_DEVICES; we'll set it after reading config

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
    # load model
    model = CreateModel(args)

    # find checkpoint: prefer newest .ckpt in model_save_path if any
    model_dir = getattr(args, 'model_save_path', './save_model')
    ckpt_candidates = glob.glob(os.path.join(model_dir, '*.ckpt'))
    if not ckpt_candidates:
        raise FileNotFoundError(f"No checkpoint (.ckpt) found in {model_dir}")
    # pick most recently modified
    ckpt_path = max(ckpt_candidates, key=os.path.getmtime)

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

    ds_test = SegData(dataname="cov19",#cov19
                    csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=test_workers)

    trainer = pl.Trainer(accelerator=accelerator, devices=devices)
    model.eval()
    trainer.test(model, dl_test)
