import random

from pathlib import Path
from torch.utils.data import DataLoader

from nirpredict.data.dataset import DeadTreeDataset, NIRDataset
from nirpredict.data.sampler import BalancedSampler
from nirpredict.data.image_processing import get_image_processor

from nirpredict.utils.augment import Augmentations
from nirpredict.utils.datautils import load_and_organize_data, stratify_images_by_patch_count


def prepare_datasets(conf):
    hdf5_path = Path(conf.data_folder).parent / conf.hdf5_file

    image_patch_map = load_and_organize_data(hdf5_path)

    train_keys, val_keys, test_keys = stratify_images_by_patch_count(image_patch_map, conf.val_size, conf.test_size)

    random.seed(None) # makes loader non-deterministic

    train_transform = Augmentations()
    val_transform = None
    test_transform = None

    image_processor = get_image_processor(conf.model, conf.backbone)

    train_dataset = DeadTreeDataset(
        hdf5_file=hdf5_path,
        keys=train_keys,
        crop_size=conf.train_crop_size,
        transform=train_transform,
        image_processor=image_processor,
    )
    val_dataset = DeadTreeDataset(
        hdf5_file=hdf5_path,
        keys=val_keys,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
    )
    test_dataset = DeadTreeDataset(
        hdf5_file=hdf5_path,
        keys=test_keys,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
    )

    train_loader = DataLoader(train_dataset, batch_size=conf.train_batch_size, sampler=BalancedSampler(hdf5_path, train_keys), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=conf.val_batch_size, sampler=BalancedSampler(hdf5_path, val_keys), shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=conf.test_batch_size, sampler=BalancedSampler(hdf5_path, test_keys), shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader


def load_data(conf):
    train_dataset, val_dataset, test_dataset = prepare_datasets(conf)

    train_nir_dataset = NIRDataset(train_dataset.dataset)
    val_nir_dataset = NIRDataset(val_dataset.dataset)
    test_nir_dataset = NIRDataset(test_dataset.dataset)

    train_nir_loader = DataLoader(train_nir_dataset, batch_size=conf.train_batch_size, shuffle=True)
    val_nir_loader = DataLoader(val_nir_dataset, batch_size=conf.val_batch_size, shuffle=False)
    test_nir_loader = DataLoader(test_nir_dataset, batch_size=conf.test_batch_size, shuffle=False)

    return train_nir_loader, val_nir_loader, test_nir_loader
