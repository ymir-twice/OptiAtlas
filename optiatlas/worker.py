"""Workers are separated from the dependency-light planning command."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .paths import configure_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["train-generator", "train-sft", "train-rl", "generate"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configure_imports()
    cfg = json.loads(args.config.read_text())
    if args.task == "train-sft":
        from .released_data import install_sft_loader
        install_sft_loader()
        from optics_agent_sft.training.runner import run_training
        run_training(args.config)
    elif args.task == "train-generator":
        from draw_align_sft.runner import run
        run(cfg)
    elif args.task == "train-rl":
        from omegaconf import OmegaConf
        from verl.trainer.main_ppo import main as official_main
        official_main.__wrapped__(OmegaConf.create(cfg))
    else:
        from .inference import install_release_provider
        install_release_provider()
        from optics_experiments.benchmark_launch import run
        run(cfg)


if __name__ == "__main__":
    main()
