from typing import Any, Dict, Optional, Tuple, List
from types import SimpleNamespace

import torch
import torchvision
from torchvision.datasets import MNIST, ImageFolder, FashionMNIST
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import numpy as np

from .texture_datamodule import TextureDataModule
from .utils import transfer_color, otsu_threshold, fuse
from .utils import unify_images_intensities, add_smooth_intensity_to_masks
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, fg_images, bg_images, masks,
                 bg_bg_labels, bg_fg_labels,
                 fg_labels, transform=None, target_transform=None):
        self.fg_images, self.bg_images, self.masks, self.bg_bg_labels, self.bg_fg_labels, self.fg_labels = \
            fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, fg_labels
        self.transform = transform

    def __len__(self):
        return len(self.fg_images)

    def __getitem__(self, index):
        fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, fg_labels = \
            self.fg_images[index], self.bg_images[index], self.masks[index], self.bg_bg_labels[index], \
            self.bg_fg_labels[
                index], self.fg_labels[index]
        if self.transform:
            fg_images = self.transform(fg_images.float())
            bg_images = self.transform(bg_images.float())
            masks = self.transform(masks.float())
        return fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, fg_labels


class ForegroundTextureDataModule():
    def __init__(
            self,
            data_dir: str = "data/",
            dataset_type: str = 'MNIST',
            im_size: List[int] = [64, 64],
            random_resizing_shifting: bool = False,
            train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
            batch_size: int = 64,
            unify_fg_objects_intensity=False,
            transforms=None
    ) -> None:

        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.im_size = im_size
        self.random_resizing_shifting = random_resizing_shifting
        supported_datasets = ['MNIST', 'FashionMNIST', 'dsprites']
        if dataset_type not in supported_datasets:
            raise ValueError(f'{dataset_type} is not supported. The supported datasets are {supported_datasets}')
        self.dataset_type = dataset_type
        self.unify_fg_objects_intensity = unify_fg_objects_intensity
        self.train_val_test_split = train_val_test_split
        self.prepare_data()
        self.setup()

    def return_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    def prepare_data(self) -> None:
        if self.dataset_type == 'MNIST':
            trainset = MNIST(self.data_dir, train=True, download=True, transform=self.transforms)
            testset = MNIST(self.data_dir, train=False, download=True, transform=self.transforms)
        elif self.dataset_type == 'FashionMNIST':
            trainset = FashionMNIST(self.data_dir, train=True, download=True, transform=self.transforms)
            testset = FashionMNIST(self.data_dir, train=False, download=True, transform=self.transforms)
        elif self.dataset_type == 'dsprites':
            trainset, testset = self.get_dsprites_datasets()
        else:
            raise ValueError(f'{self.dataset_type} is not supported')
        self.fg_images = torch.concat((trainset.data, testset.data)).unsqueeze(1)
        self.fg_labels = torch.concat((trainset.targets, testset.targets))

        texture_dataset = TextureDataModule(2 * len(self.fg_images), self.data_dir,
                                            self.im_size,
                                            self.train_val_test_split, self.batch_size)
        texture_dataset.prepare_data()
        self.bg_images = texture_dataset.x
        self.bg_bg_labels = texture_dataset.y
        self.process_images()

    def setup(self):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            total_samples = self.fg_images.shape[0]
            train_percentage, val_percentage, _ = self.train_val_test_split
            train_size = int(train_percentage * total_samples)
            val_size = int(val_percentage * total_samples)
            test_size = total_samples - train_size - val_size

            dataset = CustomDataset(self.fg_images, self.bg_images, self.masks,
                                    self.bg_bg_labels, self.bg_fg_labels,
                                    self.fg_labels, transform=self.transforms)

            self.data_train, self.data_val, self.data_test = CustomDataset(*dataset[:train_size]), \
                CustomDataset(*dataset[train_size:train_size + val_size]), \
                CustomDataset(*dataset[train_size + val_size:])

    def _resize_fg(self):
        def get_uniform():
            return (.4 - .6) * torch.rand((1,)).item() + 1

        images_ = torch.zeros_like(self.fg_images)
        for i in range(self.fg_images.shape[0]):
            fg_size = get_uniform()
            fg = self.fg_images[i]
            x_resize = int(fg_size * fg.shape[1])
            y_resize = int(fg_size * fg.shape[2])
            tmp = torch.ones(fg.shape, dtype=torch.uint8)
            resize = torchvision.transforms.Resize((x_resize, y_resize))
            fg = resize(fg)
            random_x = torch.randint(0, self.fg_images[i].shape[1] - fg.shape[1], (1,)).item()
            random_y = torch.randint(0, self.fg_images[i].shape[2] - fg.shape[2], (1,)).item()
            tmp[0, random_x:random_x + fg.shape[1],
            random_y:random_y + fg.shape[2]] = fg
            images_[i] = tmp
        self.fg_images = images_

    def process_fg_images(self):
        self.fg_images = F.resize(self.fg_images, size=self.im_size,
                                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                  antialias=False)
        if self.random_resizing_shifting:
            self._resize_fg()
        self.masks, thresholds = otsu_threshold(self.fg_images)
        if self.unify_fg_objects_intensity:
            if self.dataset_type == 'FashionMNIST' or self.dataset_type == 'MNIST':
                self.fg_images = unify_images_intensities(self.fg_images, thresholds, target_min=200, target_max=250)
            elif self.dataset_type == 'dsprites':
                self.fg_images = add_smooth_intensity_to_masks(self.fg_images,
                                                               torch.randint(2, 100,
                                                                             size=(self.fg_images.shape[0], 1, 1, 1)))
                self.fg_images = unify_images_intensities(self.fg_images, thresholds, target_min=150, target_max=250)
            else:
                raise RuntimeError(
                    f'the parameters of `unify_images_intensities` is not setup for dataset:{self.dataset_type}')

    def _blend_foreground_with_background(self, fg_images: torch.Tensor, bg_images: torch.Tensor,
                                          fg_masks: torch.Tensor):
        fg_images_blended = torch.zeros_like(fg_images, dtype=fg_images.dtype)
        for i in range(len(fg_images)):
            fg_images_blended[i] = fuse(bg_images[i], fg_images[i], fg_masks[i].to(torch.uint8))
        return fg_images_blended

    def color_images(self, images: torch.Tensor, reference_path: str) -> torch.Tensor:
        transformer = transforms.Compose([transforms.Resize(self.im_size),
                                          transforms.ToTensor()])
        dataset = ImageFolder(reference_path, transform=transformer)
        tensor_list = [img for img, _ in dataset]
        reference_images = torch.stack(tensor_list)
        images = torch.repeat_interleave(images[:, None, ...], repeats=3, dim=1).to(reference_images.dtype)
        for idx, image in enumerate(images):
            images[idx] = transfer_color(image, reference_images[torch.randint(0, len(reference_images), (1,))[0]])
        return images

    def process_images(self):
        # process the fg images and produce masks.
        self.process_fg_images()
        # blend fg with bg.
        self.fg_images = self._blend_foreground_with_background(self.fg_images,
                                                                self.bg_images[:self.bg_images.shape[0] // 2],
                                                                self.masks)
        self.bg_fg_labels = self.bg_bg_labels[:self.bg_images.shape[0] // 2]
        self.bg_bg_labels = self.bg_bg_labels[self.bg_images.shape[0] // 2:]
        self.bg_images = self.bg_images[self.bg_images.shape[0] // 2:]

    def get_dsprites_datasets(self) -> Tuple[SimpleNamespace, SimpleNamespace]:
        dataset_zip = np.load(self.data_dir + 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        imgs = torch.tensor(dataset_zip['imgs'])
        # that would be the class of each fg object
        latents_values = torch.tensor(dataset_zip['latents_values'][:, 1])
        shuffled_indices = torch.randint(0, len(imgs), size=(len(imgs),))
        imgs = imgs[shuffled_indices]
        latents_values = latents_values[shuffled_indices]
        # take only 70K images to be consistent with other datasets
        imgs = imgs[:70000]
        latents_values = latents_values[:70000]
        # latents_classes = dataset_zip['latents_classes']
        # a dummy data division into training and testing just to keep the API consistent.
        trainset, testset = SimpleNamespace(), SimpleNamespace()
        trainset.data = imgs[:imgs.shape[0] // 2]
        testset.data = imgs[imgs.shape[0] // 2:]
        trainset.targets = latents_values[:latents_values.shape[0] // 2]
        testset.targets = latents_values[latents_values.shape[0] // 2:]
        return trainset, testset

    def train_dataloader(self) -> DataLoader[Any]:

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
        )
