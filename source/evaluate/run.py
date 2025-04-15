import hydra
import mutinfo
import numpy

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import trange

import yaml

import bebeziana


def ndarray_representer(dumper: yaml.Dumper, array: numpy.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())


@hydra.main(version_base=None, config_path="./config.d", config_name="config")
def run_test(config : DictConfig) -> None:
    try:
        # Switched to reseeding each run for better reproducibility.
        #bebeziana.seed_everything(config["seed"], to_be_seeded=config["to_be_seeded"])

        # Resolving some parts of the config and storing them separately for later post-processing.
        setup = {}
        setup["estimator"]    = OmegaConf.to_container(config["estimator"], resolve=True)
        setup["distribution"] = OmegaConf.to_container(config["distribution"], resolve=True)
        
        setup["n_samples"] = int(config["n_samples"])
        setup["n_runs"]    = int(config["n_runs"])

        # Results for post-processing.
        results = {}
        results["mutual_information"] = {"values": []}

        for index in trange(config["n_runs"]):
            bebeziana.seed_everything(config["seed"] + index, to_be_seeded=config["to_be_seeded"])

            estimator       = instantiate(config["estimator"], _convert_="object")
            random_variable = instantiate(config["distribution"], _convert_="object")

            x, y = random_variable.rvs(config["n_samples"])
            results["mutual_information"]["values"].append(estimator(x, y))

        # All values.
        results["mutual_information"]["values"] = numpy.array(results["mutual_information"]["values"])

        # Mean and standard deviation (might be unstable).
        results["mutual_information"]["mean"]   = float(numpy.mean(results["mutual_information"]["values"]))
        results["mutual_information"]["std"]    = float(numpy.std(results["mutual_information"]["values"]))

        # Meduian and interquartile range (more robust to the outliers).
        results["mutual_information"]["median"] = float(numpy.median(results["mutual_information"]["values"]))
        results["mutual_information"]["half_interquartile_range"] = float(
            numpy.quantile(results["mutual_information"]["values"], 0.75) -
            numpy.quantile(results["mutual_information"]["values"], 0.25)
        )

        path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        with open(path / "setup.yaml", 'w') as file:
            yaml.dump(setup, file, default_flow_style=False)

        with open(path / "results.yaml", 'w') as file:
            yaml.dump(results, file, default_flow_style=False)

    except Exception as exception:
        print(exception) # Just ignoring exceptions so the sweeping continues.



if __name__ == "__main__":
    yaml.add_representer(numpy.ndarray, ndarray_representer)
    run_test()
