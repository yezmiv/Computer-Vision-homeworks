import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader    
import pandas as pd
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms

from torchvision.models import MobileNet_V3_Small_Weights
from data import get_transforms, seed as SEED

ROOT= Path(__file__).resolve().parent.parent   
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"

class PetDataset(Dataset):
    def __init__(self, csv_path, transform): #read the CSV save the transform
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): 
        row = self.df.iloc[idx] #get one row from csv
        img_path = ROOT / row["path"] 
        label = int(row["class_idx"])
        img = Image.open(img_path).convert("RGB")
        if self.transform: #apply the image trsnsformations
            img = self.transform(img)
        return img, label

def make_loaders(batch_size=32):
    train_tf, eval_tf = get_transforms()
    train_dataset = PetDataset(SPLITS_DIR / "train.csv", transform=train_tf)
    val_dataset = PetDataset(SPLITS_DIR / "val.csv", transform=eval_tf)
    test_dataset = PetDataset(SPLITS_DIR / "test.csv", transform=eval_tf)

    g = torch.Generator()
    g.manual_seed(SEED)  #shuffle del train riproducibile
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

#train without augmentation (for clean train accuracy and for features)
def make_train_eval_loader(batch_size=32):
    _, eval_tf = get_transforms()
    ds = PetDataset(SPLITS_DIR / "train.csv", transform=eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_loaders()

    print(f"Train loader: {len(train_loader)} batch from {train_loader.batch_size}")
    print(f"Val loader:  {len(val_loader)} batch from {val_loader.batch_size}")
    print(f"Test loader:  {len(test_loader)} batch from {test_loader.batch_size}")

    #verify the shapes
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")





