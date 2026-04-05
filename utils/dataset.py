import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from monai.transforms import (Compose, EnsureChannelFirstd,
                              NormalizeIntensityd, RandGaussianNoised,
                              RandRotated, RandZoomd, Resized, ToTensord)
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, '.hf_cache')
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', HF_CACHE_DIR)
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(HF_CACHE_DIR, 'hub'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(HF_CACHE_DIR, 'hub'))
os.environ.setdefault('HF_MODULES_CACHE', os.path.join(HF_CACHE_DIR, 'modules'))

import transformers.dynamic_module_utils as dynamic_module_utils
import transformers.utils.hub as transformers_hub

dynamic_module_utils.HF_MODULES_CACHE = os.path.join(HF_CACHE_DIR, 'modules')
transformers_hub.TRANSFORMERS_CACHE = os.path.join(HF_CACHE_DIR, 'hub')
os.makedirs(dynamic_module_utils.HF_MODULES_CACHE, exist_ok=True)
os.makedirs(transformers_hub.TRANSFORMERS_CACHE, exist_ok=True)

from transformers import AutoTokenizer

try:
    import pywt
except ImportError:
    pywt = None


class SegData(Dataset):

    def __init__(self, dataname, csv_path=None, root_path=None, tokenizer=None,
                 mode='train', image_size=[224, 224], wavelet_type='haar',
                 auto_prompt_from_mask=False):

        super(SegData, self).__init__()

        self.dataname = dataname
        self.mode = mode
        self.root_path = root_path
        self.wavelet_type = wavelet_type
        self.auto_prompt_from_mask = auto_prompt_from_mask
        self.data = pd.read_csv(csv_path, sep=None, engine='python')

        self.data_format = self._detect_data_format()
        self.text_column = self._detect_text_column()
        self.study_slice_ranges = self._build_study_slice_ranges()
        self.records = self._build_records()
        self.image_size = image_size
        self.transforms = self.transform(self.image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample = self.records[idx]

        if self.data_format in ('prepared_2d', 'paired_png'):
            base_image = self._load_grayscale(sample['image_path'])
            image, image2 = self._create_wavelet_pair(base_image)
            gt = self._load_grayscale(sample['mask_path'])
        else:
            image = self._load_grayscale(sample['image_path'])
            image2 = self._load_grayscale(sample['image2_path'])
            gt = self._load_grayscale(sample['mask_path'])

        caption = sample['caption']
        if self.auto_prompt_from_mask and self.data_format in ('prepared_2d', 'paired_png'):
            caption = self._build_prompt_from_mask(sample, gt)

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24,
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = self.transforms({'image': image, 'image2': image2, 'gt': gt})

        image, image2, gt = data['image'], data['image2'], data['gt']
        gt = (gt > 0).float()
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)}

        return ([image,image2, text], gt)

    def _detect_data_format(self):
        columns = set(self.data.columns)
        if {'image_path', 'mask_path'}.issubset(columns):
            return 'prepared_2d'
        if 'Image' in columns:
            frames_dir = os.path.join(self.root_path, 'frames')
            masks_dir = os.path.join(self.root_path, 'masks')
            if os.path.isdir(frames_dir) and os.path.isdir(masks_dir):
                return 'paired_png'
            return 'legacy'
        raise ValueError(
            f"Unsupported dataset format for {self.root_path}. "
            f"Available columns: {sorted(columns)}"
        )

    def _detect_text_column(self):
        for column in ('Description', 'prompt', 'text'):
            if column in self.data.columns:
                return column
        raise ValueError(
            f"Could not find a text column in {sorted(self.data.columns)}. "
            "Expected one of: Description, prompt, text."
        )

    def _build_records(self):
        records = []
        if self.data_format == 'prepared_2d':
            for row in self.data.to_dict('records'):
                records.append({
                    'image_path': self._resolve_path(row['image_path']),
                    'mask_path': self._resolve_path(row['mask_path']),
                    'caption': str(row.get(self.text_column, '')),
                    'study_id': row.get('study_id'),
                    'slice_idx': row.get('slice_idx'),
                })
            return records

        if self.data_format == 'paired_png':
            for row in self.data.to_dict('records'):
                image_name = row['Image']
                records.append({
                    'image_path': os.path.join(self.root_path, 'frames', image_name),
                    'mask_path': os.path.join(self.root_path, 'masks', image_name),
                    'caption': str(row.get(self.text_column, '')),
                })
            return records

        for row in self.data.to_dict('records'):
            mask_name = row['Image']
            records.append({
                'image_path': self._resolve_legacy_image_path(mask_name, 'Images_H'),
                'image2_path': self._resolve_legacy_image_path(mask_name, 'Images_L'),
                'mask_path': os.path.join(self.root_path, 'GTs', mask_name),
                'caption': str(row.get(self.text_column, '')),
            })
        return records

    def _resolve_path(self, path_value):
        path_value = str(path_value)
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(self.root_path, path_value))

    def _resolve_legacy_image_path(self, mask_name, image_dir):
        candidates = [mask_name]
        if mask_name.startswith('mask_'):
            candidates.insert(0, mask_name.replace('mask_', '', 1))
        for candidate in candidates:
            path = os.path.join(self.root_path, image_dir, candidate)
            if os.path.isfile(path):
                return path
        return os.path.join(self.root_path, image_dir, candidates[0])

    def _load_grayscale(self, path):
        return np.array(Image.open(path).convert('L'))

    def _normalize_wavelet(self, array):
        array = array.astype(np.float32)
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value > min_value:
            array = (array - min_value) / (max_value - min_value) * 255.0
        else:
            array = np.zeros_like(array, dtype=np.float32)
        return array.astype(np.uint8)

    def _create_wavelet_pair(self, image):
        if pywt is not None:
            low_freq, (freq_lh, freq_hl, freq_hh) = pywt.dwt2(image, self.wavelet_type)
        else:
            if self.wavelet_type != 'haar':
                raise ImportError(
                    "pywt is not installed. Only the built-in 'haar' fallback is available."
                )
            low_freq, (freq_lh, freq_hl, freq_hh) = self._haar_dwt2(image)
        low_freq = self._normalize_wavelet(low_freq)
        high_freq = (
            self._normalize_wavelet(freq_lh).astype(np.float32)
            + self._normalize_wavelet(freq_hl).astype(np.float32)
            + self._normalize_wavelet(freq_hh).astype(np.float32)
        )
        high_freq = self._normalize_wavelet(high_freq)
        return high_freq, low_freq

    def _haar_dwt2(self, image):
        image = image.astype(np.float32)
        pad_h = image.shape[0] % 2
        pad_w = image.shape[1] % 2
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')

        top_left = image[0::2, 0::2]
        top_right = image[0::2, 1::2]
        bottom_left = image[1::2, 0::2]
        bottom_right = image[1::2, 1::2]

        low_freq = (top_left + top_right + bottom_left + bottom_right) / 4.0
        freq_lh = (top_left - top_right + bottom_left - bottom_right) / 4.0
        freq_hl = (top_left + top_right - bottom_left - bottom_right) / 4.0
        freq_hh = (top_left - top_right - bottom_left + bottom_right) / 4.0
        return low_freq, (freq_lh, freq_hl, freq_hh)

    def _build_study_slice_ranges(self):
        if {'study_id', 'slice_idx'}.issubset(self.data.columns):
            grouped = self.data.groupby('study_id')['slice_idx']
            return {
                study_id: (int(slice_indices.min()), int(slice_indices.max()))
                for study_id, slice_indices in grouped
            }
        return {}

    def _build_prompt_from_mask(self, sample, mask):
        lesion = mask > 0
        lesion_pixels = int(lesion.sum())
        if lesion_pixels == 0:
            return "No visible COVID-19 lesion in this chest CT slice."

        height, width = lesion.shape
        area_ratio = lesion_pixels / float(height * width)
        left_pixels = int(lesion[:, : width // 2].sum())
        right_pixels = int(lesion[:, width // 2 :].sum())

        if left_pixels > 0 and right_pixels > 0:
            laterality = "bilateral lungs"
        elif left_pixels > 0:
            laterality = "left lung"
        else:
            laterality = "right lung"

        if area_ratio < 0.01:
            extent = "small"
        elif area_ratio < 0.05:
            extent = "moderate"
        else:
            extent = "extensive"

        slice_level = ""
        study_id = sample.get('study_id')
        slice_idx = sample.get('slice_idx')
        if study_id in self.study_slice_ranges and slice_idx is not None:
            min_idx, max_idx = self.study_slice_ranges[study_id]
            if max_idx > min_idx:
                relative_idx = (float(slice_idx) - min_idx) / (max_idx - min_idx)
                if relative_idx < 0.33:
                    slice_level = "upper"
                elif relative_idx < 0.66:
                    slice_level = "middle"
                else:
                    slice_level = "lower"

        if slice_level:
            return f"{extent.capitalize()} COVID-19 lesion in the {laterality} on the {slice_level} chest CT slice."
        return f"{extent.capitalize()} COVID-19 lesion in the {laterality} on this chest CT slice."

    def estimate_pos_weight(self, max_samples=None, min_value=1.0, max_value=256.0):
        records = self.records if max_samples is None else self.records[:max_samples]
        pos_pixels = 0
        neg_pixels = 0
        for sample in records:
            gt = self._load_grayscale(sample['mask_path']) > 0
            pos_pixels += int(gt.sum())
            neg_pixels += int(gt.size - gt.sum())

        if pos_pixels <= 0:
            return float(min_value)

        pos_weight = neg_pixels / pos_pixels
        pos_weight = max(float(min_value), min(float(max_value), float(pos_weight)))
        return pos_weight

    def transform(self,image_size=[224,224]):

        keys = ["image","image2", "gt"]
        if self.mode == 'train':  
            trans = Compose([
                EnsureChannelFirstd(keys=keys),  
                RandZoomd(keys=keys, min_zoom=0.95, max_zoom=1.15, mode=["bicubic","bicubic", "nearest"], prob=0.3),  
                RandRotated(keys=keys, range_x=[-0.3, 0.3], keep_size=True, mode=['bicubic','bicubic', 'nearest'],  prob=0.3),  
                RandGaussianNoised(keys=["image2"], prob=0.3, mean=0.0, std=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["image2"],spatial_size=image_size,mode='bicubic'),
                Resized(keys=["gt"], spatial_size=image_size, mode='nearest'), 
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                NormalizeIntensityd(['image2'], channel_wise=True),
                ToTensord(keys=["image","image2", "gt"])
            ])            
        else:  
            trans = Compose([
                EnsureChannelFirstd(["image","image2","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["image2"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                NormalizeIntensityd(['image2'], channel_wise=True),
                ToTensord(["image","image2","gt"]),
            ])

        return trans
