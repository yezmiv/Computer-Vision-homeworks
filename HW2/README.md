# Homework 2: Image Classification on Oxford-IIIT Pet

This project uses PyTorch and transfer learning to classify images from Oxford-IIIT Pet dataset (37 breeds) using MobileNetV3-Small (ImageNet-1K pre-trained), is used to compare three different approaches.

## Environment
- Python 3.14.3
- torch 2.15.0
- torchvision 0.29.2
- scikit-learn 1.9.1
- numpy 2.4.4
- pandas 3.0.2
- matplotlib 3.10.8
- seaborn 0.13.2
- pillow 12.2.0
- tqdm 4.70.1

## Device

The device is selected automatically:
- CUDA if an NVIDIA GPU is available.
- MPS if an Apple Silicon GPU is available.
- CPU otherwise.

## Dataset
The Oxford-IIIT Pet (Parkhi et al., CVPR 2012) contains images of 37 cat and dog breeds.
* License: CC BY-SA 4.0.
* Source: https://www.robots.ox.ac.uk/~vgg/data/pets/
* The dataset is downloaded automatically by `torchvision.datasets.OxfordIIITPet` on first run (approximately 800 MB). 
* It is not included in this repository.

# Data splits
The official `trainval`partition is divided in training and validation sets using stratified 80/20. 
* `train`: 2,944 images 
* `val` 736 images
* `test`: 3,669 images kept untouched for the final evaluation

Split files are saved in `splits/` as CSV files with path, class_idx, split and class_name.

Random Seed = 42 (Python, NumPy, PyTorch).

## Approaches
Three approaches are implemented: 
1. `full`: full fine-tuning of MobileNetV3-Small (all the 1,555,781 params are trainable).
2. `head`: freeze the convolutional features and train only the classifier (628,773 trainable params).
3. `feat`: use the frozen network as feature extractor (576-dimensional vectors) followed by scikit-learn classifiers (LinearSVC, LogisticRegression), with StandardScaler fit on training only.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Reproduce

Prepare the data splits splits, downloads dataset on first run:
```bash
python3 src/data.py
```

Run all three approaches:
```bash
python3 homework2.py --approach all
```

Or run them individually:
```bash
python3 homework2.py --approach head
python3 homework2.py --approach full
python3 homework2.py --approach feat
```

Optional flags:

```bash
#custom number of epochs and learning rate for full/head
python homework2.py --approach head --epochs 10 --lr 0.001

#run final evaluation on the test set after training
python homework2.py --approach feat --evaluate
```

Evaluate saved checkpoints on the test set (without retraining):
```bash
python src/evaluate.py
```

## Results
The evaluation results, including accuracy, macro-F1 and confusion matrices, are saved in `results/`:
- `final_table.json`: contains all metrics
- `confusion_full.png`, `confusion_head.png`: contains the confusion matrices
- `head_history.json`, `full_history.json`: contains training curves
- `feat_results.json`: contains feature-extraction results

Model checkpoints are saved in `checkpoints/`:
- `full_best.pth` and `head_best.pth`
.

The detailed experimental analysis is available in `report_HW2.pdf`.

## Repository structure
* homework2.py — main script to run the project.
* src/ — contains the fondamental modules:
    * data.py — downloads the dataset and creates stratified data splits.
    * dataset.py — defines the PetDataset class and creates DataLoaders.
    * model.py — defines MobileNetV3-Small for the three approaches.
    * training.py — contains the training loop for full and head.
    * features.py — extracts image features and trains scikit-learn classifiers.
    * evaluate.py — evaluates the models on the test set and generates confusion matrices.
* splits/ — contains the train, validation and test CSV files, and class counts.
* checkpoints/ — stores the best model weights for full and head.
* results/ — contains evaluation metrics in JSON format and confusion matrix images in PNG format.
* requirements.txt — lists the Python dependencies.
* report_HW2.pdf — project report.
```
