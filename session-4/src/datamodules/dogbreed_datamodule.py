import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as pl
from torchvision.datasets.utils import extract_archive, download_file_from_google_drive

class DogBreedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all breed folders
        self.breeds = sorted(os.listdir(root_dir))  # Get breed names from folders
        # Create breed to index mapping
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        
        # Get all image paths and their labels
        self.image_paths = []
        self.labels = []
        for breed in self.breeds:
            breed_path = os.path.join(root_dir, breed)
            for img_name in os.listdir(breed_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(breed_path, img_name))
                    self.labels.append(self.breed_to_idx[breed])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DogBreedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        try:
            print('Prepping data for dog breed')
            dataset_url = os.getenv('DATASET_URL')
            file_id = dataset_url.split('=')[-1]
            filename = os.getenv('DATASET_FILENAME')
            root = self.data_dir
            download_file_from_google_drive(file_id, root, filename)
            fpath = os.path.join(root, filename) # this is what download_file_from_google_drive does
            ## extract downloaded dataset
            print(f'FPATH FOR DATASET >>> {fpath}')
            from_path = os.path.expanduser(fpath)
            extract_archive(from_path)
        except Exception as e:
            import traceback
            print(f'failed to prep {traceback.format_exc()}')



    def setup(self, stage=None):
        # Called on every GPU
        print(f'setup called...dataa dir: {self.data_dir}')
        print(f'train list dir : {os.listdir(os.path.join(self.data_dir, "train"))}')
        print(f'test list dir : {os.listdir(os.path.join(self.data_dir, "test"))}')
        
        self.train_dataset = DogBreedDataset(
            os.path.join(self.data_dir, 'train'),
            transform=self.transform
        )
        self.val_dataset = DogBreedDataset(
            os.path.join(self.data_dir, 'val'),
            transform=self.transform
        )
        self.test_dataset = DogBreedDataset(
            os.path.join(self.data_dir, 'test'),
            transform=self.transform
        )
        
    def prepare_data(self):
        pass
        # try:
        #     print('Prepping data for dog breed')
        #     dataset_url = os.getenv('DATASET_URL')
        #     file_id = dataset_url.split('=')[-1]
        #     filename = os.getenv('DATASET_FILENAME')
        #     root = self._dl_path
        #     download_file_from_google_drive(file_id, root, filename)
        #     # download_and_extract_archive(
        #     #     url = os.getenv('DATASET_URL'),
        #     #     # url="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
        #     #     download_root=self._dl_path,
        #     #     remove_finished=True
        #     # )
        #     fpath = os.path.join(root, filename) # this is what download_file_from_google_drive does
        #     ## extract downloaded dataset
        #     print(f'FPATH FOR DATASET >>> {fpath}')
        #     from_path = os.path.expanduser(fpath)
        #     extract_archive(from_path)
        # except Exception as e:
        #     import traceback
        #     print(f'failed to prep {traceback.format_exc()}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=4)
    
    def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                         num_workers=4)