import h5py
import torch

import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset

from nirpredict.data.image_processing import apply_image_processor


class DeadTreeDataset(Dataset):
    def __init__(self, hdf5_file, keys, crop_size=256, transform=None, image_processor=None):
        self.hdf5_file = hdf5_file
        self.keys = keys
        self.crop_size = crop_size
        self.transform = transform
        self.image_processor = image_processor

        self._adjust_image_processor_mean_std() # Fix incompatable image mean/std to support four channels

    def _adjust_image_processor_mean_std(self):
        if self.image_processor:
            if len(self.image_processor.image_mean) == 3:
                self.image_processor.image_mean.insert(0, 0.5)
            if len(self.image_processor.image_std) == 3:
                self.image_processor.image_std.insert(0, 0.5)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):            
        image, label = self._load_data(idx)
        image, label = self._preprocess_image_and_label(image, label)

        return image, label

    def _load_data(self, idx):
        key = self.keys[idx]

        with h5py.File(self.hdf5_file, "r") as hf:
            image = hf[key]['image'][()]
            label = hf[key]['label'][()]

        return image, label

    def _preprocess_image_and_label(self, image, label):
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        image = image / 255.0

        image = image.permute(2, 0, 1)  # Convert to (C, H, W) format
        label = label.unsqueeze(0)  # Convert to (1, H, W) format

        image, label = self._center_crop_or_pad(image, label, self.crop_size)

        if self.image_processor:
            image, label = apply_image_processor(image, label, self.image_processor)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def _center_crop_or_pad(self, image, label, size=256):
        h, w = image.shape[1:]  # image is in (C, H, W) format

        pad_h, pad_w = max(size - h, 0), max(size - w, 0)

        if pad_h > 0 or pad_w > 0:
            image, label = self._pad_image_and_label(image, label, pad_h, pad_w)
        
        return self._crop_center(image, label, size)
        
    def _pad_image_and_label(self, image, label, pad_h, pad_w):
        image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0)
        label = F.pad(label, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0)

        return image, label
    
    def _crop_center(self, image, label, size):
        h, w = image.shape[1:]
        x, y = (w - size) // 2, (h - size) // 2

        return image[:, y:y + size, x:x + size], label[:, y:y + size, x:x + size]


class NIRDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data = self.original_dataset[idx]

        if isinstance(data, tuple) and len(data) == 2:
            image, label = data
        else:
            raise ValueError(f"Expected a tuple of (image, label) at index {idx}, but got {type(data)}")

        if image.shape[0] != 4:
            raise ValueError(f"Expected image with 4 channels, but got {image.shape[0]} channels.")

        nir_target = image[0, :, :]
        rgb_input = image[1:, :, :]

        return rgb_input, nir_target
