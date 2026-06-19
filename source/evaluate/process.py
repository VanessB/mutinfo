import bebeziana
import itertools
import pandas
import polars
import re
import yaml

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


distribution_names = {
    "CorrelatedNormal":    "Correlated Normal",
    "CorrelatedUniform":   "Correlated Unform",
    "CorrelatedStudent":   "Correlated Student",
    "LogGammaExponential": "Log-Gamma-Exp.",
    "SmoothedUniform":     "Smoothed Uniform",
    "UniformlyQuantized":  "Uniformly Quantized",
}


def MAE(x, first, second):
    return (x[first] - x[second]).abs()

def postprocess_table(table: str) -> str:
    """
    Oh god, why is polars.to_latex so bad?..
    """

    sub_patterns = {
        "priority":  [re.compile(r"_\d+_"), ""],
        #"priority":  [re.compile(r"\\_\d+\\_"), ""],
        "cline":     [re.compile(r"cline"), "cmidrule"],
        "midbottom": [re.compile(r"\\cmidrule\{\d+-\d+\}\s+\\bottomrule"), r"\\bottomrule"],
        #"header":   [re.compile(r"\\toprule.*\\\\(?P<values>)[\s&\\]*(?P<columns>)[\s&\\]*\\midrule")]
    }

    for pattern in sub_patterns.values():
        table = re.sub(pattern[0], pattern[1], table)

    return table


if __name__ == "__main__":
    ignore_parts = [".ipynb_checkpoints"]

    configs_path = Path("./config.d/tables")
    for config_path in configs_path.rglob("*.yaml"):
        if not config_path.is_file():
            continue

        ignore = False
        for part in ignore_parts:
            ignore = ignore or part in config_path.parts
        if ignore:
            continue

        table_config = instantiate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True), _convert_="object")
        table_name   = table_config["name"]
        #table_name   = config_path.stem

        print(table_name)
        
        final_data = []
        for source_name, source_config in table_config["sources"].items():
            # Read data.
            directory_path = Path(source_config["path"])
            data_path = directory_path / "data.csv"
            if data_path.exists():
                data = polars.read_csv(data_path)
            else:
                data = bebeziana.read(directory_path, ["setup.yaml", "results.yaml"])
                data.to_csv(data_path)

            # Pinning columns' values.
            for column, value in itertools.chain(
                table_config["pin"].items(),
                source_config["pin"].items()
            ):
                data = data.filter(polars.col(column) == value)

            # Saving source and test name.
            data = data.with_columns(
                Source = polars.lit(f"_{source_config['priority']}_{source_name}"),
                Distribution = data["name.distribution"].replace(distribution_names)
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

            final_data.append(data)


        output_path = Path("./tables") / (table_name + ".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_data = polars.concat(final_data).sort(table_config["rows_to_chart"])
        final_data.write_csv(output_path)

        # Great tables do not support multirow cells.
        #table = (
        #    GT(final_data)
        #    # Round floats to 2 decimals
        #    .fmt_number(columns=[col for col, dtype in final_data.schema.items() if dtype in (polars.Float64, polars.Float32)], decimals=2)
        #    # Replace missing values with "--"
        #    .sub_missing(missing_text="--")
        #).as_latex()

        final_data = final_data.to_pandas(use_pyarrow_extension_array=False)
        final_data = final_data.set_index(table_config["rows_to_chart"])
        
        table = final_data.style \
            .format(
                na_rep="--",
                formatter="$ {:0.2f} $".format
            ) \
            .to_latex(
                hrules=True,
                clines="skip-last;data",
                column_format='l'*final_data.index.nlevels + 'c'*len(final_data.columns),
                multicol_align='c',
                multirow_align='l',
            )
        table = postprocess_table(table)

        output_path = Path("./tables") / (table_name + ".tex")
        
        with open(output_path, 'w') as table_file:
            table_file.write(table)