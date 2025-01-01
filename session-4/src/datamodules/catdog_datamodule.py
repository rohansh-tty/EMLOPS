from pathlib import Path
from typing import Union
import os
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive, extract_archive
from dotenv import load_dotenv

load_dotenv()

class CatDogImageDataModule(L.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        """Download images and prepare images datasets."""
        try:
            dataset_url = os.getenv('DATASET_URL')
            file_id = dataset_url.split('=')[-1]
            filename = os.getenv('DATASET_FILENAME')
            root = self._dl_path
            download_file_from_google_drive(file_id, root, filename)
            # download_and_extract_archive(
            #     url = os.getenv('DATASET_URL'),
            #     # url="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
            #     download_root=self._dl_path,
            #     remove_finished=True
            # )
            fpath = os.path.join(root, filename) # this is what download_file_from_google_drive does
            ## extract downloaded dataset
            print(f'FPATH FOR DATASET >>> {fpath}')
            from_path = os.path.expanduser(fpath)
            extract_archive(from_path)
            
            
            # validation and test data 
            test_dataset_url = os.getenv('TEST_DATASET_URL')
            test_file_id = test_dataset_url.split("=")[-1]
            test_filename = os.getenv('TEST_DATASET_FILENAME')
            test_root = self.data_path.joinpath("test")
            download_file_from_google_drive(test_file_id, test_root, test_filename)
            test_fpath = os.path.join(test_root, test_filename)
            print(f'FPATH FOR TEST DATASET >>> {test_fpath}')
            from_path = os.path.expanduser(fpath)
            test_from_path = os.path.expanduser(test_fpath)
            extract_archive(test_from_path)
        except Exception as e:
            import traceback 
            print(f'failed to prep data, {traceback.format_exc()}')
    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation/test loaders."""
        if train:
            dataset = self.create_dataset(self.data_path.joinpath(""), self.train_transform)
            print('number of classses >>> ', dataset.classes)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("test"), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)  # Using validation dataset for testing