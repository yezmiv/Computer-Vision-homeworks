import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC #cause linearSVC is faster than SVC with linear kernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from dataset import make_loaders, make_train_eval_loader
from model import build_model
from data import set_seed, seed as seed


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def extract_feat(model, loader, device): #extract features from the model for all images in the loader
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            #forward through the convolutional part only
            feat = model.features(images)
            feat = model.avgpool(feat) #average pooling to reduce the spatial dimensions
            feat = torch.flatten(feat, 1)

            all_features.append(feat.cpu().numpy()) #from mps to cpu and then to numpy
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0) #mat numerical describtion of the image 
    y = np.concatenate(all_labels, axis=0) 
    return X, y


def run_feat(batch_size=32): #pipeline for the feat approach
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    model = build_model("feat").to(DEVICE)
    _, val_loader, test_loader = make_loaders(batch_size=batch_size)
    train_loader = make_train_eval_loader(batch_size=batch_size)
    
    #extract features
    t0 = time.time()
    print("Extract train features.")
    X_train, y_train = extract_feat(model, train_loader, DEVICE)
    print("Extrac val features")
    X_val, y_val = extract_feat(model, val_loader, DEVICE)
    print("Extrac test features")
    X_test, y_test = extract_feat(model, test_loader, DEVICE)
    extract_time = time.time() - t0

    print(f"Train features: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Extraction time: {extract_time:.1f}s")

   
    scaler = StandardScaler()  #standardize
    X_train = scaler.fit_transform(X_train) #impara media e deviazione standard 
    X_val = scaler.transform(X_val) #same trasformazione
    X_test = scaler.transform(X_test)

    #classifiers (train and evaluate) 
    results = {
        "approach": "feat",
        "n_trainable_params": 0,
        "extraction_time_s": extract_time,
        "feature_dim": X_train.shape[1],
        "classifiers": {},
    }

    classifiers = { #usa solo c=1 di default
        "LinearSVC": LinearSVC(max_iter=5000),
        "LogisticRegression": LogisticRegression(max_iter=2000),
    }

    for name, clf in classifiers.items():
        print(f"\nTraining {name}")
        t1 = time.time()
        clf.fit(X_train, y_train) #train the classifier
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        fit_time = time.time() - t1 #claculate the training time
        #predict validation and test label
        val_pred = clf.predict(X_val) 
        test_pred = clf.predict(X_test)

        #save LR test predictions for confusion matrix
        if name == "LogisticRegression":
            np.save(RESULTS_DIR / "featLR_test_pred.npy", test_pred)
            np.save(RESULTS_DIR / "feat_test.npy", y_test)

        #the accuracy 
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")  #check if the model performs well on all classes, not only the most common ones

        print(f"{name}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f},  test_f1={test_f1:.4f},  (fit {fit_time:.1f}s)")

        results["classifiers"][name] = {
            "fit_time_s": fit_time,
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "test_f1_macro": float(test_f1),
            "train_acc": float(train_acc),
        }


    out_path = RESULTS_DIR / "feat_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    return results


if __name__ == "__main__":
    run_feat()