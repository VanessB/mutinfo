import argparse
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

def RMAE(x, first, second):
    return (x[second] / x[first] - 1.0).abs()

def format_uncertainty(x, value, uncertainty):
    value, uncertainty = x[value], x[uncertainty]
    
    is_valid = uncertainty > 0

    # log10(uncertainty) floor + 1 matches the 2 significant figures rule for error
    decimals = (
        (- (polars.when(is_valid).then(uncertainty).otherwise(1.0).log10().floor()).cast(polars.Int32) + 1)
        .clip(lower_bound=0) # Negative decimals (large uncertainties) map to 0 for string formatting
    )

    # Create a temporary DataFrame to process rows by their specific decimal precision
    df = polars.DataFrame({
        "v": value,
        "u": uncertainty,
        "is_valid": is_valid
    }).with_columns(
        polars.when(polars.col("is_valid"))
        .then(decimals)
        .otherwise(0)
        .alias("decimals")
    )

    # Construct the formatted string dynamically for each decimal group
    return (
        df.with_columns(
            polars.struct(["v", "u", "decimals", "is_valid"])
            .map_elements(
                lambda row: (
                    f"${row['v']:.{row['decimals']}f} \\pm {row['u']:.{row['decimals']}f}$"
                    if row["is_valid"]
                    else f"${row['v']} \\pm {row['u']}$"
                ),
                return_dtype=polars.String
            )
            .alias("formatted")
        )
        .get_column("formatted")
    )

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

def table_to_latex(
    table: polars.DataFrame,
    index_columns: list[str],
    formatter=None,
) -> str:
    """
    Converts a Polars DataFrame to a formatted LaTeX table via Pandas.
    """

    # Conversion to Pandas is necessary for the .style API
    table = table.to_pandas(use_pyarrow_extension_array=False)
    table = table.set_index(index_columns)

    # Default formatter.
    if formatter is None:
        formatter={column: '${:.2f}$' for column in table.select_dtypes(include='float').columns}

    try:
        table_latex = table.style \
            .format(
                na_rep="--",
                formatter=formatter
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
        return table_latex.to_string() # Fallback to string


def table_to_markdown(
    table: polars.DataFrame,
    index_columns: list[str],
) -> str:
    """
    Converts a Polars DataFrame to a formatted Markdown table via Pandas.
    """

    # Conversion to Pandas is necessary for the .style API
    table = table.to_pandas(use_pyarrow_extension_array=False)
    table = table.set_index(index_columns)

    try:
        table_markdown = table.to_markdown()
          
        return table_markdown

    except Exception as e:
        logger.error(f"Markdown conversion failed: {e}")
        return table.to_string() # Fallback to string
    

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
    save: bool=True,
    use_cache: bool=True,
) -> polars.DataFrame:
    file_path = path / name

    if use_cache and file_path.exists():
        return polars.read_parquet(file_path)
    
    # Read and cache source from individual files if parquet doesn't exist.
    data = polars.from_pandas(bebeziana.read(path, config_files))
    if save:
        logger.info(f"Generating a parquet file at {path}")
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
        table_config["rows_to_chart"] + [table_config["column_to_chart"]["name"]] + list(source_config["aggregate"])
    ).agg(polars.mean(table_config["outputs"]))

    # Aggregation.
    data = data.group_by(
        table_config["rows_to_chart"] + [table_config["column_to_chart"]["name"]]
    ).agg(polars.all().min_by(table_config["aggregation"]["by"]))

    # Calculating postprocessing targets.
    if "postprocessing" in table_config.keys():
        data = data.with_columns(
            **{target_config["name"]: target_config["function"](data) for target_config in table_config["postprocessing"]}
        )

    data = data.drop(set(source_config["aggregate"]) - {table_config["aggregation"]["output"]}) # Be careful not to drop the column we want to output.

    if "apply" in table_config["column_to_chart"].keys():
        data = data.with_columns(
            polars.col(table_config["column_to_chart"]["name"]).map_elements(table_config["column_to_chart"]["apply"])
        )

    data = data.pivot(
        table_config["column_to_chart"]["name"],
        index=table_config["rows_to_chart"],
        values=table_config["aggregation"]["output"],
        sort_columns=True
    )

    return data

def create_table(table_config: dict[str, Any], use_cache: bool=True) -> polars.DataFrame:
    sources_data = []
    
    for source_name, source_config in table_config["sources"].items():
        logger.info(f"Processing source: {source_name}")
        
        data = load_source(Path(source_config["path"]), use_cache=use_cache)
        processed = process_source(data, source_name, source_config, table_config)
        sources_data.append(processed)
  
    return polars.concat(sources_data, how="diagonal").sort(table_config["rows_to_chart"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Results Processor',
        description='This program reads, aggregates and outpus experimental data as formatted tables. The tables are described using .yaml config files.',
        epilog='MUTINFO'
    )

    parser.add_argument("-c", "--config", type=Path, help="configs path", default="./config.d/tables")
    parser.add_argument("-o", "--output", type=Path, help="output path", default="./tables")

    parser.add_argument("-r", "--recache", action='store_true', help="regenerate cache files", default=False)

    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    
    try:
        configs = load_configs(arguments.config)
    except Exception as exception:
        logger.error(f"Failed to load configs: {exception}")
        exit(1)
    
    for table_config in configs:
        table_name = table_config["name"]
        logger.info(f"Creating table: {table_name}")
  
        try:
            table = create_table(table_config, not arguments.recache)
  
            # Save CSV
            csv_path = arguments.output / f"{table_name}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            table.write_csv(csv_path) # Fixed: changed file_path to csv_path
  
            # Save LaTeX
            latex_path = arguments.output / f"{table_name}.tex"
            latex_path.parent.mkdir(parents=True, exist_ok=True)
            latex_table = table_to_latex(table, table_config["rows_to_chart"])
            with open(latex_path, 'w') as f:
                f.write(latex_table)

            # Save Markdown
            markdown_path = arguments.output / f"{table_name}.md"
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_table = table_to_markdown(table, table_config["rows_to_chart"])
            with open(markdown_path, 'w') as f:
                f.write(markdown_table)
                
            logger.info(f"Successfully saved {table_name}")

        except Exception as exception:
            logger.error(f"Error processing table {table_name}: {exception}")