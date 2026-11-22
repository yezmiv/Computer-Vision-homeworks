import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from PIL import Image
from dataset import make_loaders
from model import build_model, count_trainable_params
from data import set_seed

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def evaluate_on_test(model, loader, device): #run the model on the test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path): #compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(14, 12))

    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120) #save in png
    plt.close()

#how many error 
def top_confused_pairs(y_true, y_pred, class_names, k=5):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names))) #confusion matrix
    np.fill_diagonal(cm, 0) #azzerala diagonale perche mi interessano solo gli errori 
    order = np.argsort(cm.ravel())[::-1][:k] #ordina gli errori dalpiu grande al piu piccolo
    rows, cols = np.unravel_index(order, cm.shape)
    #return the most confused class pairs
    return [{"true": class_names[r], "pred": class_names[c], "errors": int(cm[r, c])}
            for r, c in zip(rows, cols)]

#show the ex of prediction correct and wrong  
def plot_prediction_examples(test_df, y_true, y_pred, class_names, save_path, n=4, seed=42):
    rng = np.random.default_rng(seed)
    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]
    #randomly select ex from both groups
    groups = [("correct", rng.choice(correct, n, replace=False), "green"),
              ("wrong", rng.choice(wrong, n, replace=False), "red")]
    
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.6)) #figure with two row of img
    for r, (_, idxs, color) in enumerate(groups):
        for c, i in enumerate(idxs):
            img = Image.open(ROOT / test_df.iloc[i]["path"]).convert("RGB") 
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            axes[r, c].set_title(f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}",
                                 fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def run_evaluation(batch_size=32): # evaluate full/head on test set
    #get class names i need them for CM labels
    set_seed()
    train_loader, val_loader, test_loader = make_loaders(batch_size=batch_size) 
    class_df = train_loader.dataset.df.drop_duplicates("class_idx").sort_values("class_idx")
    class_names = class_df["class_name"].tolist()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_results = {} #store all final results

    
    
    test_df = test_loader.dataset.df 

    def analyse(name, y_true, y_pred):
        plot_confusion_matrix(y_true, y_pred, class_names, f"Confusion Matrix: {name}", RESULTS_DIR / f"confusion_{name}.png")
        plot_prediction_examples(test_df, y_true, y_pred, class_names, RESULTS_DIR / f"examples_{name}.png")
        return top_confused_pairs(y_true, y_pred, class_names)
    
    for approach in ["full", "head"]:
        print(f"\nEvaluating {approach} on test set: ")

        model = build_model(approach).to(DEVICE) #create the model
        ckpt_path = CHECKPOINTS_DIR / f"{approach}_best.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)) #load the best model wights
        n_params = count_trainable_params(model)

        #evaluate
        y_true, y_pred = evaluate_on_test(model, test_loader, DEVICE)

        #metrics acc
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"{approach}: test_acc={acc:.4f}  test_f1={f1:.4f}")

        #Analyse predictions and generate evaluation plots
        top_conf = analyse(approach, y_true, y_pred)

        hist_path = RESULTS_DIR / f"{approach}_history.json"
        with open(hist_path) as f:
            hist = json.load(f)

        final_results[approach] = {
            "test_acc": float(acc),
            "test_f1_macro": float(f1),
            "n_trainable_params": n_params,
            "top_confused": top_conf,
            "train_acc": hist["train_acc_clean"],
            "val_acc": hist["best_val_acc"],
            "time_s": hist["total_time_s"],
            "lr": hist["lr"],
            "best_epoch": hist["best_epoch"],
        }

    #i load FEAT results that are already computed
    feat_path = RESULTS_DIR / "feat_results.json"
    with open(feat_path, "r") as f:
        feat_data = json.load(f)

    for clf_name, clf_res in feat_data["classifiers"].items():
        key = f"feat_{clf_name}" #create a name to classifier 
       
        #add classifier result
        final_results[key] = { 
            "test_acc": clf_res["test_acc"],
            "test_f1_macro": clf_res["test_f1_macro"],
            "n_trainable_params": 0, #no trainable neural network parameters
            "fit_time_s": clf_res["fit_time_s"],
            "time_s": feat_data["extraction_time_s"] + clf_res["fit_time_s"], 
             "val_acc": clf_res["val_acc"],
        }

    #confusion matrix for feat LogisticRegression
    feat_pred_path = RESULTS_DIR / "featLR_test_pred.npy"
    feat_y_path = RESULTS_DIR / "feat_test.npy" 
    if feat_pred_path.exists() and feat_y_path.exists():
            y_pred_feat = np.load(feat_pred_path)
            y_true_feat = np.load(feat_y_path)
            final_results["feat_LogisticRegression"]["top_confused"] = analyse("featLR", y_true_feat, y_pred_feat)
        
    print("\nFINAL TABLE:")
    print(f"{'Approach':<25} {'test_acc':>10} {'test_f1':>10} {'params':>12}")
    for name, res in final_results.items():
        print(f"{name:<25} {res['test_acc']:>10.4f} {res['test_f1_macro']:>10.4f} "
              f"{res['n_trainable_params']:>12,}")

    #save table
    out_path = RESULTS_DIR / "final_table.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    return final_results


if __name__ == "__main__":
    run_evaluation()