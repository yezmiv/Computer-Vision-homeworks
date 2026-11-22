import argparse
import sys
from pathlib import Path

#make "src" importable when running from HW2/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from training import train_model
from features import run_feat
from evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run HW2 approaches")
    parser.add_argument("--approach", choices=["full", "head", "feat", "all"], required=True)
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for full/head (default 15).")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for full/head (default:0.0001).")
    parser.add_argument("--evaluate", action="store_true", help="Also run evaluate.py at the end.")
    args = parser.parse_args()

    if args.approach in ("head", "all"):
       train_model("head", num_epochs=args.epochs, lr=args.lr)
    if args.approach in ("full", "all"):
       train_model("full", num_epochs=args.epochs, lr=args.lr)
    if args.approach in ("feat", "all"):
        run_feat()

    if args.evaluate:
        run_evaluation()


if __name__ == "__main__":
    main()