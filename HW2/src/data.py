import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision import transforms
from sklearn.model_selection import train_test_split

#CONFIGURATION BASE

#path to the root of the project
ROOT = Path(__file__).resolve().parent.parent #cartella HW2/
DATA_DIR = ROOT / "data" #cartella HW2/data/
SPLITS_DIR = ROOT / "splits" #cartella HW2/splits/

seed = 42
val_split = 0.2 #proportion of training used for validation set


#fix random seeds for reproducibility
def set_seed(seed: int = seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

def clean_class_name(raw_name: str) -> str: #fatto perche nella tabella mi esce da 0-36
    name = raw_name.rsplit("_", 1)[0]
    return name.replace("_", " ").title()

# Funzione principale: scarica e prepara gli split
def prepare_splits(force_redownload: bool = False) -> None:
    """
    Scarica il dataset, crea train/val/test stratificati,
    salva i CSV in splits/, stampa la tabella per razza.
    """
    set_seed(seed)

    # Scarica il trainval ufficiale (per ora senza transform: ci serve solo la lista)
    trainval_full = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        download=True,          # scarica solo se manca
        transform=None,         # non ci interessa processare le immagini qui
    )

    # Scarica il test ufficiale (separato!)
    test_full = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        download=True,
        transform=None,
    )



    #create a DataFrame with image paths and labels
    def dataset_to_df(ds, split_name: str) -> pd.DataFrame:
        rows = [] #Usa gli attributi interni effettivamente esistenti _images e _labels di OxfordIIITPet."""

        for path, class_idx in zip(ds._images, ds._labels):
            rows.append({"path": str(Path(path).relative_to(ROOT)), "class_idx": class_idx, "split": split_name}) #cambiato perche mi dava il path lungo e non coerente
        return pd.DataFrame(rows)


    #separate tables for trainval and test
    df_trainval = dataset_to_df(trainval_full, "trainval")
    df_test = dataset_to_df(test_full, "test")



    # Split trainval into train and validation
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_split,
        random_state=seed,
        stratify=df_trainval["class_idx"]
    )

    df_train = df_train.assign(split="train")
    df_val = df_val.assign(split="val")
    df_test = df_test.assign(split="test")

    #-- Nomi puliti delle razze
    class_names_clean = [clean_class_name(c) for c in trainval_full.classes]

    # Aggiungi la colonna class_name a tutti e tre i DataFrame
    for df in (df_train, df_val, df_test):
        df["class_name"] = df["class_idx"].map(lambda i: class_names_clean[i])
    #--

    #CSV files for each split
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(SPLITS_DIR / "train.csv", index=False)
    df_val.to_csv(SPLITS_DIR / "val.csv", index=False)
    df_test.to_csv(SPLITS_DIR / "test.csv", index=False)

    #count images per class in each split
    combined = pd.concat(
    [df_train[["class_name", "split"]],
     df_val[["class_name", "split"]],
     df_test[["class_name", "split"]]],
    ignore_index=True)

    table = (combined.groupby(["class_name", "split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0))

    table = table.reindex(columns=["train", "val", "test"], fill_value=0)
    table["total"] = table.sum(axis=1)

    print(table)
    print("\nTotal images:")
    print(table.sum())
    table.to_csv(SPLITS_DIR / "class_counts.csv")

# Image transformations
def get_transforms():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    eval_tf = weights.transforms()

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
        saturation=0.2), transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)
    ])

    return train_tf, eval_tf

if __name__ == "__main__":
    prepare_splits()
    print("\nSplit pronti.")