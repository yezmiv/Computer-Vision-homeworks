import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import make_loaders
from model import build_model, count_trainable_params
from data import set_seed
from dataset import make_loaders, make_train_eval_loader

#in this i will implement the training only for full and head approach, feat is just for feature extraction so no training is needed
ROOT= Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR= ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

#implement the training loop for one epoch
def train_one_epoch(model, loader, optimizer, criterion, device, freeze_backbone=False):
    model.train() #put the model in training mode
   
    if freeze_backbone:  #head: i BatchNorm del backbone restano in eval
        model.features.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward + step
        optimizer.zero_grad() #clear gradients
        loss.backward() #compute gradients
        optimizer.step() #update weights

        #statistiche
        running_loss += loss.item() * images.size(0) #sum of losses for the batch
        _, predicted = torch.max(outputs, dim=1) #find the class with the highest score (_, discard the max value and predicted is the index of the class with the highest score)
        running_correct += (predicted == labels).sum().item() #confront predicted with labels and sum the number of correct predictions
        total += images.size(0)

    avg_loss = running_loss / total #sum of losses / total number of samples so we can habe an avg loss
    avg_acc  = running_correct / total
    return avg_loss, avg_acc

def evaluate(model, loader, criterion, device):
    model.eval() #put the model in evaluation mode
    
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad(): #no need to compute gradients for evaluation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            running_correct += (predicted == labels).sum().item()
            total += images.size(0)

    avg_loss = running_loss / total
    avg_acc  = running_correct / total
    return avg_loss, avg_acc


def train_model(approach, num_epochs=15, batch_size=32, lr=0.0001):
    set_seed()
    train_loader, val_loader, _ = make_loaders(batch_size=batch_size) #create the data loaders

    #build model and move to the device cpu mps or gpu
    model = build_model(approach).to(DEVICE)
    n_trainable = count_trainable_params(model)

    criterion = nn.CrossEntropyLoss() #loss function and optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )  #just pass trainable params to the optimizer (frozen ones are skipped)

    #the output 
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    #history one dict per epoch
    history = []
    best_val_acc = 0.0
    best_epoch = -1
    total_start = time.time()

    #loop over epochs 
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                        freeze_backbone=(approach == "head")) #here we train the model
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE) #and here we misure the performance 
        epoch_time = time.time() - epoch_start #duration of the epoch

        
        print(f"Epoch {epoch:2d}/{num_epochs} "
              f"train_loss={train_loss:.3f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.4f}  "
              f"({epoch_time:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_s": epoch_time,
        })

        #here i want to save the best model based on validation accuracy 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path = CHECKPOINTS_DIR / f"{approach}_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"the new best (val_acc={val_acc:.4f}) saved in {checkpoint_path.name}")

    
    total_time = time.time() - total_start #total training time

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    _, train_acc_clean = evaluate(model, make_train_eval_loader(batch_size), criterion, DEVICE)
    print(f"Clean train accuracy (no augmentation): {train_acc_clean:.4f}") 

    
    #save history as json (with config + final results)
    history_path = RESULTS_DIR / f"{approach}_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "approach": approach,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_trainable_params": n_trainable,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "total_time_s": total_time,
            "history": history,
            "seed": 42,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "train_acc_clean": train_acc_clean,

        }, f, indent=2)

    print(f"\nbest_epoch={best_epoch}, best_val_acc={best_val_acc:.4f} and "
          f"total_time={total_time:.1f}s" 
          
          )

    return best_val_acc



if __name__ == "__main__":
    train_model("head", num_epochs=15, lr=0.0001)
    train_model("full", num_epochs=15, lr=0.0001)