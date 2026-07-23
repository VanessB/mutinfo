import bebeziana
import itertools
import logging
import pandas
import polars
import re
import yaml

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any

# Logger.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


DISTRIBUTION_NAMES = {
    "CorrelatedNormal":        "Correlated Normal",
    "CorrelatedStudent":       "Correlated Student",
    "CorrelatedUniform":       "Correlated Uniform",
    "LogGammaExponential":     "Log-Gamma-Exp.",
    "RareEventChannel":        "Rare Event Channel",
    "SmoothedDiscreteUniform": "Smoothed Discrete Uniform",
    "SmoothedUniform":         "Smoothed Uniform",
    "UniformlyQuantized":      "Uniformly Quantized",
}


def MAE(x, first, second):
    return (x[first] - x[second]).abs()

def postprocess_table(table: str) -> str:
    """
    Clean up pandas.to_latex output using regex.
    """

    sub_patterns = {
        "priority":  [re.compile(r"_\d+_"), ""],
        "cline":     [re.compile(r"cline"), "cmidrule"],
        "midbottom": [re.compile(r"\\cmidrule\{\d+-\d+\}\s+\\bottomrule"), r"\\bottomrule"],
    }

    for pattern_pair in sub_patterns.values():
        table = re.sub(pattern_pair[0], pattern_pair[1], table)

    return table

def table_to_latex(table: polars.DataFrame, index_columns: list[str]) -> str:
    """
    Converts a Polars DataFrame to a formatted LaTeX table via Pandas.
    """

    # Conversion to Pandas is necessary for the .style API
    table = table.to_pandas(use_pyarrow_extension_array=False)
    table = table.set_index(index_columns)

    try:
        table_latex = table.style \
            .format(
                na_rep="--",
                formatter="$ {:0.2f} $".format
            ) \
            .to_latex(
                hrules=True,
                clines="skip-last;data",
                column_format='l'*table.index.nlevels + 'c'*len(table.columns),
                multicol_align='c',
                multirow_align='l',
            )
          
        return postprocess_table(table_latex)

    except Exception as e:
        logger.error(f"LaTeX conversion failed: {e}")
        return pd_df.to_string() # Fallback to string
    

def load_configs(path: Path, ignore_parts: list[str]=[".ipynb_checkpoints"]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Invalid configs path: {path}")
      
    configs = []
    for config_path in path.rglob("*.yaml"):
        if not config_path.is_file() or any(part in config_path.parts for part in ignore_parts):
            continue
  
        # Load and resolve OmegaConf to a standard Python dictionary
        container = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        configs.append(instantiate(container, _convert_="object"))
  
    return configs

def load_source(
    path: Path,
    name: str="data.parquet",
    config_files: list[str]=["setup.yaml", "results.yaml"],
    save: bool=True
) -> polars.DataFrame:
    file_path = path / name
  
    if file_path.exists():
        return polars.read_parquet(file_path)
    
    # Read and cache source from individual files if parquet doesn't exist.
    data = polars.from_pandas(bebeziana.read(path, config_files))
    if save:
        data.write_parquet(file_path)

    return data

def process_source(
    data: polars.DataFrame,
    source_name: str,
    source_config: dict,
    table_config: dict
) -> polars.DataFrame:
    # Pinning columns' values.
    filters = [
        polars.col(column) == value for column, value
            in itertools.chain(table_config["pin"].items(), source_config["pin"].items())
    ]

    if filters:
        data = data.filter(*filters)

    # Saving source and test name.
    data = data.with_columns(
        Source = polars.lit(f"_{source_config['priority']}_{source_name}"),
        Distribution = polars.col("name.distribution").replace(DISTRIBUTION_NAMES)
    )
    
    # Calculating targets.
    data = data.with_columns(
        **{target_config["name"]: target_config["function"](data) for target_config in table_config["targets"]}
    )

    # Pre-aggregation averaging.
    data = data.group_by(
        table_config["rows_to_chart"] + [table_config["column_to_chart"]["name"]] + source_config["aggregate"]
    ).agg(polars.mean(table_config["outputs"]))

    # Aggregation.
    data = data.group_by(
        table_config["rows_to_chart"] + [table_config["column_to_chart"]["name"]]
    ).agg(polars.all().min_by(table_config["aggregation"]["by"]))

    if "apply" in table_config["column_to_chart"].keys():
        data = data.with_columns(
            polars.col(table_config["column_to_chart"]["name"]).map_elements(table_config["column_to_chart"]["apply"])
        ).drop(source_config["aggregate"])

    data = data.pivot(
        table_config["column_to_chart"]["name"],
        index=table_config["rows_to_chart"],
        values=table_config["aggregation"]["output"],
        sort_columns=True
    )

    return data

def create_table(table_config: dict[str, Any]) -> polars.DataFrame:
    sources_data = []
    
    for source_name, source_config in table_config["sources"].items():
        logger.info(f"Processing source: {source_name}")
        
        data = load_source(Path(source_config["path"]))
        processed = process_source(data, source_name, source_config, table_config)
        sources_data.append(processed)
  
    return polars.concat(sources_data).sort(table_config["rows_to_chart"])


if __name__ == "__main__":
    configs_path = Path("./config.d/tables")
    output_path  = Path("./tables")

    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        configs = load_configs(configs_path)
    except Exception as exception:
        logger.error(f"Failed to load configs: {exception}")
        exit(1)
    
    for table_config in configs:
        table_name = table_config["name"]
        logger.info(f"Creating table: {table_name}")
  
        try:
            table = create_table(table_config)
  
            # Save CSV
            csv_path = output_path / f"{table_name}.csv"
            table.write_csv(csv_path) # Fixed: changed file_path to csv_path
  
            # Save LaTeX
            latex_path = output_path / f"{table_name}.tex"
            table_latex = table_to_latex(table, table_config["rows_to_chart"])
            with open(latex_path, 'w') as f:
                f.write(table_latex)
                
            logger.info(f"Successfully saved {table_name}")

        except Exception as exception:
            logger.error(f"Error processing table {table_name}: {exception}")