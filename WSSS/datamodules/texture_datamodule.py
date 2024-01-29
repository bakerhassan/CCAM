from typing import Any, Dict, Optional, Tuple, List
import os

import torchvision
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms import transforms


def extract_patch(im, patch_size):
    x = torch.randint(0, high=im.shape[1] - patch_size[0], size=(1,)).item()
    y = torch.randint(0, high=im.shape[2] - patch_size[1], size=(1,)).item()
    patch = im[:, x:x + patch_size[0], y:y + patch_size[1]]
    return patch


class TextureDataModule():
    def __init__(
            self,
            images_number: int,
            data_dir: str = "data/",
            im_size: List[int] = [64, 64],
            train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
            batch_size: int = 64,
    ) -> None:

        self.images_number = images_number
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.texture_dir = os.path.join(current_script_dir, "textures")
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.texture_sources: str = None
        self.batch_size = batch_size
        self.im_size = im_size
        self.x = None
        self.y = None
        self.train_val_test_split = train_val_test_split
        self.prepare_data()
        self.setup()

    def return_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


    def prepare_data(self) -> None:
        self.texture_sources, self.texture_sources_labels = self.load_textures()
        self.process_bg_images()

    def setup(self) -> None:
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = TensorDataset(self.x, self.y)
            total_samples = len(dataset)
            train_percentage, val_percentage, _ = self.train_val_test_split
            train_size = int(train_percentage * total_samples)
            val_size = int(val_percentage * total_samples)
            test_size = total_samples - train_size - val_size

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

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

    def load_textures(self):
        """ Load source textures

        Load the texture images that will be used as a background for the MNIST
        digits.

        Args
        ----
        root: str
            Path to the folder with texture images.

        Returns
        -------
        textures: torch.Tensor
            A tensor containing texture images.
        ids: list[int]
            A list with the id of each texture image.

        Notes
        -----
        The name of each texture image file is supposed to have a
        unique numeric id. The id of the `i`-th image is saved in `ids[i]`.
        """

        textures, ids = [], []

        # Define a transform to convert PIL images to tensors
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        for filename in os.listdir(self.texture_dir):
            path = os.path.join(self.texture_dir, filename)
            if os.path.isfile(path) and filename.lower().endswith('.tif'):
                # Read the TIFF image using PIL
                with Image.open(path) as im:
                    # Convert the image to grayscale
                    im = im.convert('L')
                    # Convert PIL image to a tensor
                    tensor_image = torchvision.transforms.functional.pil_to_tensor(im)  # Add a batch dimension
                    textures.append(tensor_image)
                    ids.append(int(os.path.splitext(filename)[0][1:]))  # Extract id from filename

        # Stack the tensor images along a new dimension to create a single tensor
        textures = torch.stack(textures)
        ids = torch.tensor(ids)

        return textures, ids

    def process_bg_images(self):
        textures = torch.zeros((self.images_number, 1, self.im_size[0], self.im_size[1]),
                               dtype=torch.uint8)
        texture_labels = torch.zeros(self.images_number, dtype=torch.uint8)
        for i in range(self.images_number):
            associated_texture_label = torch.randint(0, len(self.texture_sources_labels), size=(1,))[0]
            textures[i] = extract_patch(self.texture_sources[associated_texture_label], self.im_size)
            texture_labels[i] = associated_texture_label
        self.x = textures
        self.y = texture_labels
