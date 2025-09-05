import hydra
import mutinfo
import numpy
import traceback

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
        if "raw" in config.keys():
            setup["raw"] = OmegaConf.to_container(config["distribution"], resolve=True)
        
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

        # If the estimator is parametrized, get the total number of parameters.
        # TODO: make pretty.
        #_est = estimator
        #if isinstance(_est, TransformedMutualInformationEstimator):
        #    _est = _est.estimator
        #if isinstance(_est, ParametricMutualInformationEstimator):
        #    setup["n_parameters"] = _est.count_parameters(x, y)
        setup["n_parameters"] = 0
        if "parameters_counter" in config:
            parameters_counter = instantiate(config["parameters_counter"], _convert_="object")
            setup["n_parameters"] = parameters_counter(estimator, x, y)

        for statistic in ["mutual_information"]:
            # All values.
            results[statistic]["values"] = numpy.array(results[statistic]["values"])
    
            # Mean and standard deviation (might be unstable).
            results[statistic]["mean"]   = float(numpy.mean(results[statistic]["values"]))
            results[statistic]["std"]    = float(numpy.std(results[statistic]["values"]))
    
            # Meduian and interquartile range (more robust to outliers).
            results[statistic]["median"] = float(numpy.median(results[statistic]["values"]))
            results[statistic]["lower_quartile"] = float(numpy.quantile(results[statistic]["values"], 0.25))
            results[statistic]["upper_quartile"] = float(numpy.quantile(results[statistic]["values"], 0.75))
            results[statistic]["half_interquartile_range"] = results[statistic]["upper_quartile"] - results[statistic]["lower_quartile"]

        path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        with open(path / "setup.yaml", 'w') as file:
            yaml.dump(setup, file, default_flow_style=False)

        with open(path / "results.yaml", 'w') as file:
            yaml.dump(results, file, default_flow_style=False)

    except Exception as exception:
        traceback.print_exc() # Just ignoring exceptions so sweeping continues.



if __name__ == "__main__":
    yaml.add_representer(numpy.ndarray, ndarray_representer)
    run_test()
