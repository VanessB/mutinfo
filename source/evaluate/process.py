#import bebeziana
import yaml
import pandas
import re

from pathlib import Path

distribution_names = {
    "mutinfo.distributions.base.CorrelatedNormal":    "Correlated Normal",
    "mutinfo.distributions.base.CorrelatedUniform":   "Correlated Unform",
    "mutinfo.distributions.base.CorrelatedStudent":   "Correlated Student",
    "mutinfo.distributions.base.LogGammaExponential": "Log-Gamma-Exp.",
    "mutinfo.distributions.base.SmoothedUniform":     "Smoothed Uniform",
    "mutinfo.distributions.base.UniformlyQuantized":  "Uniformly Quantized",
}

estimator_data_paths = {
    "KSG":     Path("./outputs/KSG.2025.04.1"),
    "WKL":     Path("./outputs/WKL.2025.04.1"),
    "MINE":    Path("./outputs/MINE.2025.04.3"),
    "MINDE-c": Path("./outputs/MINDE.2025"),
    "MINDE-j": Path("./outputs/MINDE.2025"),
}

tables = {
    "GT_MAE": {
        "target": {
            "function": lambda x : (x["distribution.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Ground-truth Mutual Information",
        },
        "column_to_chart": "distribution.mutual_information",
        "rows_to_chart": [
            "Distribution",
            "Estimator",
        ],
        "fixed_columns": {"n_samples": 10000},
        "estimators": {
            "KSG": {
                "min_columns": ["estimator.k_neighbors"],
                "fixed_columns": {},
                "priority": 0,
            },
            "WKL": {
                "min_columns": ["estimator.k_neighbors"],
                "fixed_columns": {},
                "priority": 0,
            },
            "MINE": {
                "min_columns": ["estimator.estimator.backbone_factory.hidden_dim"],
                "fixed_columns": {"estimator.estimator.estimate_fraction": 0.5},
                "priority": 1,
            },
            "MINDE-c": {
                "min_columns": ["estimator_arch","mi_sigma"],
                "fixed_columns": {"importance_sampling": True, "estimator_type": "c"},
                "priority": 2,
            },
            "MINDE-j": {
                "min_columns": ["estimator_arch","mi_sigma"],
                "fixed_columns": {"importance_sampling": True, "estimator_type": "j"},
                "priority": 2,
            },
        },
    },
}


def postprocess_table(table: str) -> str:
    """
    Oh god, why is pandas.to_latex so bad?..
    """

    sub_patterns = {
        "priority":  [re.compile(r"_\d+_"), ""],
        "cline":     [re.compile(r"cline"), "cmidrule"],
        "midbottom": [re.compile(r"\\cmidrule\{\d+-\d+\}\s+\\bottomrule"), r"\\bottomrule"],
        #"header":   [re.compile(r"\\toprule.*\\\\(?P<values>)[\s&\\]*(?P<columns>)[\s&\\]*\\midrule")]
    }

    for pattern in sub_patterns.values():
        table = re.sub(pattern[0], pattern[1], table)

    return table


def to_latex(data: pandas.DataFrame) -> str:
    n_levels = data.index.nlevels
    n_columns = len(data.columns)

    start = r"\begin{tabular}{" + 'r'*n_levels + 'c'*n_columns + "}" + '\n' + r"    \toprule" + '\n'
    end   = r"    \bottomrule" + '\n' + r"\end{tabular}"

    header = ""
    for level_index, levels in enumerate(data.columns.levels):
        header += ' '*4 + " & " * n_levels
        for level in levels:
            header += str(level)
        header += '\n'

    return start + header + end


if __name__ == "__main__":
    data_path = Path("./outputs/")
    #data = bebeziana.read(data_path, ["setup.yaml", "results.yaml"])

    for table_name, table_config in tables.items():
        final_data = []
        for estimator_name, estimator_config in table_config["estimators"].items():            
            data = pandas.read_csv(estimator_data_paths[estimator_name] / "data.csv")

            for column, value in table_config["fixed_columns"].items():
                data = data[data[column] == value]

            for column, value in estimator_config["fixed_columns"].items():
                data = data[data[column] == value]

            data["Estimator"] = f"_{estimator_config["priority"]}_{estimator_name}"
            data["Distribution"] = data["distribution._target_"].map(distribution_names).fillna(data["distribution._target_"])

            data[table_config["target"]["name"]] = table_config["target"]["function"](data)
            data[table_config["column_to_chart"]] = data[table_config["column_to_chart"]].apply(int)

            data = pandas.DataFrame(
                data.groupby(
                    table_config["rows_to_chart"] + [table_config["column_to_chart"]] + estimator_config["min_columns"],
                    dropna=False
                )[[table_config["target"]["name"]]].mean()
            )

            data = pandas.DataFrame(
                data.groupby(
                    table_config["rows_to_chart"] + [table_config["column_to_chart"]],
                    dropna=False
                )[[table_config["target"]["name"]]].min()
            )

            data = data.unstack(table_config["column_to_chart"])

            final_data.append(data)

        
        final_data = pandas.concat(final_data).sort_index()

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

        print(table)