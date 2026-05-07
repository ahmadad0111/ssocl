"""Main function for this repo."""

import argparse
import random
import numpy as np
import torch

from utils.gpu_tools import set_gpu
from trainer.Source import SourceTrainer
from trainer.SSCL import SSCLTrainer


def set_seed(seed):
    if seed == 0:
        print("Using random seed.")
        torch.backends.cudnn.benchmark = True
        return

    print("Using manual seed:", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()

    # =========================
    # Basic parameters
    # =========================
    parser.add_argument("--model_type", type=str, default="EEGNet", choices=["EEGNet"])

    # Source data
    parser.add_argument("--source_data", type=str, default="PPB_EMO")
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--source_patience", type=int, default=30)

    # Target data
    parser.add_argument("--dataset", type=str, default="AMIGOS")

    # Training parameters
    parser.add_argument("--training_style", type=str, default="Static",
                        choices=["NEW", "COPE", "OnPro", "Baseline",
                                 "NEW_Cross", "ISOL", "AMBM", "Static"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--clip_hyper", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--update_step", type=int, default=5)

    # Memory buffer
    parser.add_argument("--buffer_size", type=int, default=200)
    parser.add_argument("--buffer_batch_size", type=int, default=32)

    # General settings
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--modality", type=str, default="ECG")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--multisource", action="store_true")

    # =========================
    # WandB settings
    # =========================
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="SSOCL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()

    # =========================
    # Device and GPU
    # =========================
    set_gpu(args.gpu)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", args.device)

    if args.use_wandb:
        import wandb

    # =========================
    # Seeds and datasets
    # =========================
    seeds = [42, 43, 44, 46, 48]

    datasets_to_run = ["AMIGOS"]

    for dataset in datasets_to_run:

        args.dataset = dataset

        for seed in seeds:

            args.seed = seed
            set_seed(args.seed)

            # =========================
            # Dataset subject range
            # =========================
            if args.dataset == "DEAP":
                args.start_range = 1
                args.subject_range = 33

            elif args.dataset == "AMIGOS":
                args.start_range = 1
                args.subject_range = 41

            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")

            # =========================
            # WandB initialization
            # =========================
            if args.use_wandb:
                run_name = args.wandb_name

                if run_name is None:
                    run_name = (
                        f"{args.source_data}_to_{args.dataset}_"
                        f"seed_{args.seed}_buffer_{args.buffer_size}"
                    )

                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=args.wandb_group,
                    name=run_name,
                    config=vars(args),
                    reinit=True
                )

            try:
                # =========================
                # Source training
                # =========================
                print(f"\nPretraining model on Source data: {args.source_data}")

                if args.use_wandb:
                    wandb.log({
                        "stage/source_training": 1,
                        "seed": args.seed,
                        "source_data": args.source_data,
                        "target_dataset": args.dataset,
                    })

                source_train = SourceTrainer(args)
                source_train.train()

                # =========================
                # Optional multisource training
                # =========================
                # if args.multisource:
                #     original_source = args.source_data
                #     args.source_data = "DEAP"
                #
                #     print(f"\nPretraining model on additional Source data: {args.source_data}")
                #
                #     source_train = SourceTrainer(args)
                #     source_train.train()
                #
                #     args.source_data = original_source

                # =========================
                # Target continual learning
                # =========================
                print(f"\nSelf-supervised continual learning on Target data: {args.dataset}")

                if args.use_wandb:
                    wandb.log({
                        "stage/target_continual_learning": 1,
                        "target_dataset": args.dataset,
                    })

                trainer = SSCLTrainer(args)
                trainer.train()

            finally:
                if args.use_wandb:
                    wandb.finish()


if __name__ == "__main__":
    main()