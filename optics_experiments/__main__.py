"""Selected benchmark worker; the historical experiment dispatcher is omitted."""
import hydra
from omegaconf import OmegaConf

@hydra.main(version_base="1.3",config_path=None,config_name=None)
def main(cfg):
    from optiatlas.paths import configure_imports
    from optiatlas.inference import install_release_provider
    configure_imports()
    install_release_provider()
    if cfg.task != "benchmark_generate":
        raise ValueError("Use python -m optiatlas for selected training and evaluation stages")
    from .benchmark_generate import run
    run(OmegaConf.to_container(cfg,resolve=True))

if __name__ == "__main__":
    main()
