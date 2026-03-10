import torch
from torch.utils.data import DataLoader
from utils.dataset import SegData
import utils.config as config
from torch.optim import lr_scheduler
from net.creratemodel import CreateModel
import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import os
import numpy as np
import random
import platform

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    print("cuda is used:",torch.cuda.is_available())
    ds_train = SegData(dataname="cov19", #cov19
                    csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='train')
    # create train dataloader
    # set num_workers conservatively on Windows to avoid spawn issues
    if platform.system().lower().startswith('win'):
        train_workers = 0
    else:
        train_workers = min(8, max(0, args.train_batch_size))
    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=train_workers)

    # decide whether to create validation dataloader
    use_val = False
    try:
        if getattr(args, 'val_csv_path', None) and os.path.isfile(args.val_csv_path):
            # basic check: val csv exists
            use_val = True
    except Exception:
        use_val = False

    model = CreateModel(args)

    if use_val:
        ds_valid = SegData(dataname="cov19",#cov19
                        csv_path=args.val_csv_path,
                        root_path=args.val_root_path,
                        tokenizer=args.bert_type,
                        image_size=args.image_size,
                        mode='valid')
        if platform.system().lower().startswith('win'):
            valid_workers = 0
        else:
            valid_workers = min(4, max(0, args.valid_batch_size))
        dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=valid_workers)

        # checkpoint & early stopping monitor validation MIoU
        model_ckpt = ModelCheckpoint(
            dirpath=args.model_save_path,
            filename=args.model_save_filename,
            monitor='val_MIoU',
            save_top_k=1,
            mode='max',
            verbose=True,
        )
        early_stopping = EarlyStopping(
            monitor='val_MIoU',
            patience=args.patience,
            mode='max',
        )
        tqdm_cb = TQDMProgressBar()
        csv_logger = CSVLogger("logs", name="train_with_val")
        # Handle device config: if it's a number (GPU ID), use 'cuda' accelerator
        if args.device == 'cpu':
            accelerator = 'cpu'
            devices = 'auto'
        else:
            accelerator = 'cuda'
            devices = [args.device] if isinstance(args.device, int) else 1
        
        trainer = pl.Trainer(logger=csv_logger,
                    min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                    accelerator=accelerator,
                    devices=devices,
                    callbacks=[model_ckpt,early_stopping, tqdm_cb],
                    enable_progress_bar=True,
                    )

        print('====start (with validation)====')
        trainer.fit(model, dl_train, dl_valid)
        print('====finish====')
    else:
        # no validation provided: train on full train set and save last checkpoint
        # when no validation, only save the last checkpoint to avoid unreliable "best" metrics
        model_ckpt = ModelCheckpoint(
            dirpath=args.model_save_path,
            filename=args.model_save_filename,
            save_top_k=0,
            verbose=True,
            save_last=True,
        )

        tqdm_cb = TQDMProgressBar()
        csv_logger = CSVLogger("logs", name="train_no_val")
        # Handle device config: if it's a number (GPU ID), use 'cuda' accelerator
        if args.device == 'cpu':
            accelerator = 'cpu'
            devices = 'auto'
        else:
            accelerator = 'cuda'
            devices = [args.device] if isinstance(args.device, int) else 1
        
        trainer = pl.Trainer(logger=csv_logger,
                            min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                            accelerator=accelerator,
                            devices=devices,
                            callbacks=[model_ckpt, tqdm_cb],
                            enable_progress_bar=True,
                            )

        print('====start (no validation)====')
        trainer.fit(model, dl_train)
        print('====finish====')

