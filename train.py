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
import re

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


def create_dataloader(args, csv_path, root_path, mode, batch_size, dataset_name, wavelet_type):
    dataset = SegData(
        dataname=dataset_name,
        csv_path=csv_path,
        root_path=root_path,
        tokenizer=args.bert_type,
        image_size=args.image_size,
        mode=mode,
        wavelet_type=wavelet_type,
        auto_prompt_from_mask=getattr(args, 'auto_prompt_from_mask', False),
    )
    if platform.system().lower().startswith('win'):
        num_workers = 0
    elif mode == 'train':
        num_workers = min(8, max(0, batch_size))
    else:
        num_workers = min(4, max(0, batch_size))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, dataloader


def resolve_resume_checkpoint(args):
    checkpoint_path = getattr(args, 'resume_checkpoint_path', None)
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if not getattr(args, 'resume_latest', False):
        return None

    model_dir = getattr(args, 'model_save_path', None)
    if not model_dir or not os.path.isdir(model_dir):
        return None

    last_candidates = []
    all_candidates = []
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if not filename.endswith('.ckpt'):
                continue
            full_path = os.path.join(root, filename)
            all_candidates.append(full_path)
            if filename.startswith('last'):
                last_candidates.append(full_path)

    candidates = last_candidates if last_candidates else all_candidates
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_metric_from_checkpoint_name(path, monitor_name):
    filename = os.path.basename(path)
    pattern = re.escape(monitor_name) + r'=([0-9]*\.?[0-9]+)'
    match = re.search(pattern, filename)
    return float(match.group(1)) if match else None


def find_best_checkpoint_file(model_dir, monitor_name, mode='max'):
    if not model_dir or not os.path.isdir(model_dir):
        return None

    scored = []
    fallback = []
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if not filename.endswith('.ckpt') or filename.startswith('last'):
                continue
            full_path = os.path.join(root, filename)
            metric_value = parse_metric_from_checkpoint_name(full_path, monitor_name)
            if metric_value is None:
                fallback.append(full_path)
            else:
                scored.append((metric_value, full_path))

    if scored:
        reverse = mode == 'max'
        scored.sort(key=lambda item: item[0], reverse=reverse)
        return scored[0][1]
    if fallback:
        return max(fallback, key=os.path.getmtime)
    return None


def write_text_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    args = get_parser()
    dataset_name = getattr(args, 'dataset_name', 'cov19')
    wavelet_type = getattr(args, 'wavelet_type', 'haar')
    seed = getattr(args, 'seed', 42)
    pl.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(getattr(args, 'matmul_precision', 'medium'))
    print("cuda is used:",torch.cuda.is_available())
    ds_train, dl_train = create_dataloader(
        args,
        args.train_csv_path,
        args.train_root_path,
        'train',
        args.train_batch_size,
        dataset_name,
        wavelet_type,
    )

    # decide whether to create validation dataloader
    use_val = False
    try:
        if getattr(args, 'val_csv_path', None) and os.path.isfile(args.val_csv_path):
            # basic check: val csv exists
            use_val = True
    except Exception:
        use_val = False

    use_test = False
    try:
        if getattr(args, 'test_csv_path', None) and os.path.isfile(args.test_csv_path):
            use_test = True
    except Exception:
        use_test = False

    if getattr(args, 'auto_pos_weight', False):
        pos_weight = ds_train.estimate_pos_weight(
            max_samples=getattr(args, 'pos_weight_max_samples', None),
            min_value=getattr(args, 'min_bce_pos_weight', 1.0),
            max_value=getattr(args, 'max_bce_pos_weight', 256.0),
        )
        args.bce_pos_weight = pos_weight
        print(f"Using BCE pos_weight={pos_weight:.4f}")
    else:
        args.bce_pos_weight = float(getattr(args, 'bce_pos_weight', 1.0))

    model = CreateModel(args)
    os.makedirs(args.model_save_path, exist_ok=True)
    log_root = getattr(args, 'log_root', 'run_logs')
    os.makedirs(log_root, exist_ok=True)
    resume_ckpt_path = resolve_resume_checkpoint(args)
    if resume_ckpt_path:
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
    trainer_kwargs = {
        "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs,
        "accelerator": 'cpu' if args.device == 'cpu' else 'cuda',
        "devices": 'auto' if args.device == 'cpu' else ([args.device] if isinstance(args.device, int) else 1),
        "enable_progress_bar": True,
        "limit_train_batches": getattr(args, 'limit_train_batches', 1.0),
        "limit_val_batches": getattr(args, 'limit_val_batches', 1.0),
        "limit_test_batches": getattr(args, 'limit_test_batches', 1.0),
        "log_every_n_steps": getattr(args, 'log_every_n_steps', 10),
        "num_sanity_val_steps": getattr(args, 'num_sanity_val_steps', 2),
        "accumulate_grad_batches": getattr(args, 'accumulate_grad_batches', 1),
    }
    precision = getattr(args, 'precision', None)
    if precision is not None:
        trainer_kwargs["precision"] = precision

    dl_test = None
    if use_test:
        _, dl_test = create_dataloader(
            args,
            args.test_csv_path,
            args.test_root_path,
            'test',
            args.valid_batch_size,
            dataset_name,
            wavelet_type,
        )

    if use_val:
        _, dl_valid = create_dataloader(
            args,
            args.val_csv_path,
            args.val_root_path,
            'valid',
            args.valid_batch_size,
            dataset_name,
            wavelet_type,
        )

        # checkpoint & early stopping monitor validation MIoU
        model_ckpt = ModelCheckpoint(
            dirpath=args.model_save_path,
            filename=args.model_save_filename,
            monitor='val_MIoU',
            save_top_k=1,
            save_last=True,
            mode='max',
            verbose=True,
        )
        early_stopping = EarlyStopping(
            monitor='val_MIoU',
            patience=args.patience,
            mode='max',
        )
        tqdm_cb = TQDMProgressBar()
        csv_logger = CSVLogger(log_root, name=getattr(args, 'experiment_name', 'train_with_val'))
        trainer = pl.Trainer(
                    logger=csv_logger,
                    callbacks=[model_ckpt,early_stopping, tqdm_cb],
                    **trainer_kwargs,
                    )

        print('====start (with validation)====')
        trainer.fit(model, dl_train, dl_valid, ckpt_path=resume_ckpt_path)

        monitor_name = model_ckpt.monitor
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if not best_ckpt_path or not os.path.isfile(best_ckpt_path):
            best_ckpt_path = find_best_checkpoint_file(args.model_save_path, monitor_name, model_ckpt.mode)

        last_ckpt_path = os.path.join(args.model_save_path, 'last.ckpt')
        summary_lines = [
            f"monitor={monitor_name}",
            f"best_model_path={best_ckpt_path}",
            f"best_model_score={trainer.checkpoint_callback.best_model_score}",
            f"last_model_path={last_ckpt_path if os.path.isfile(last_ckpt_path) else ''}",
        ]
        summary_text = '\n'.join(summary_lines) + '\n'
        write_text_file(os.path.join(args.model_save_path, 'best_checkpoint.txt'), summary_text)
        print(summary_text)

        if getattr(args, 'run_test_after_fit', True) and dl_test is not None:
            ckpt_for_test = best_ckpt_path if best_ckpt_path and os.path.isfile(best_ckpt_path) else 'last'
            print(f'====test (checkpoint={ckpt_for_test})====')
            trainer.test(model, dataloaders=dl_test, ckpt_path=ckpt_for_test)
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
        csv_logger = CSVLogger(log_root, name=getattr(args, 'experiment_name', 'train_no_val'))
        trainer = pl.Trainer(
                            logger=csv_logger,
                            callbacks=[model_ckpt, tqdm_cb],
                            **trainer_kwargs,
                            )

        print('====start (no validation)====')
        trainer.fit(model, dl_train, ckpt_path=resume_ckpt_path)
        if getattr(args, 'run_test_after_fit', True) and dl_test is not None:
            print('====test (last checkpoint)====')
            trainer.test(model, dataloaders=dl_test, ckpt_path='last')
        print('====finish====')

