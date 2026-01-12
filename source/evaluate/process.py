import bebeziana
import yaml
import pandas
import re

from pathlib import Path

distribution_names = {
    "CorrelatedNormal":    "Correlated Normal",
    "CorrelatedUniform":   "Correlated Unform",
    "CorrelatedStudent":   "Correlated Student",
    "LogGammaExponential": "Log-Gamma-Exp.",
    "SmoothedUniform":     "Smoothed Uniform",
    "UniformlyQuantized":  "Uniformly Quantized",
}

tables = {
    "lowdim_kNN_KSG_10000": {
        "data_paths": {
            "KSG":     Path("./outputs/lowdim/2025-10-20/KSG"),
            "WKL":     Path("./outputs/lowdim/2025-10-20/WKL"),
            "MINE-DV": Path("./outputs/lowdim/2025-10-20/MINE-DV"),
            "InfoNCE": Path("./outputs/lowdim/2025-10-20/InfoNCE"),
        },
        "target": {
            "function": lambda x : (x["distribution.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Number of nearest neighbours",
        },
        "column_to_chart": {
            "name": "estimator.k_neighbors",
            "apply": int,
        },
        "rows_to_chart": [
            "Distribution",
        ],
        "fixed_columns": {"n_samples": 10000},
        "estimators": {
            "KSG": {
                "min_columns": [],
                "fixed_columns": {},
                "priority": 0,
            },
        },
    },
    
    "lowdim_kNN_WKL_10000": {
        "data_paths": {
            "KSG":     Path("./outputs/lowdim/2025-10-20/KSG"),
            "WKL":     Path("./outputs/lowdim/2025-10-20/WKL"),
            "MINE-DV": Path("./outputs/lowdim/2025-10-20/MINE-DV"),
            "InfoNCE": Path("./outputs/lowdim/2025-10-20/InfoNCE"),
        },
        "target": {
            "function": lambda x : (x["distribution.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Number of nearest neighbours",
        },
        "column_to_chart": {
            "name": "estimator.k_neighbors",
            "apply": int,
        },
        "rows_to_chart": [
            "Distribution",
        ],
        "fixed_columns": {"n_samples": 10000},
        "estimators": {
            "WKL": {
                "min_columns": [],
                "fixed_columns": {},
                "priority": 0,
            },
        },
    },
    
    "lowdim_DD_MINDE-DV": {
        "data_paths": {
            "KSG":     Path("./outputs/lowdim/2025-10-20/KSG"),
            "WKL":     Path("./outputs/lowdim/2025-10-20/WKL"),
            "MINE-DV": Path("./outputs/lowdim/2025-10-20/MINE-DV"),
            "InfoNCE": Path("./outputs/lowdim/2025-10-20/InfoNCE"),
        },
        "target": {
            "function": lambda x : (x["distribution.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Number of nearest neighbours",
        },
        "column_to_chart": {
            "name": "n_samples",
            "apply": int,
        },
        "rows_to_chart": [
            "Distribution",
            "estimator.estimator.backbone_factory.hidden_dim"
        ],
        "fixed_columns": {},
        "estimators": {
            "MINE-DV": {
                "min_columns": [],
                "fixed_columns": {},
                "priority": 0,
            },
        },
    },

    "mi_estimate:base:distribution_method:GT:1k": {
        "data_paths": {
            "KSG":     Path("./outputs/lowdim/2025-10-20/KSG"),
            "WKL":     Path("./outputs/lowdim/2025-10-20/WKL"),
            "MINE-DV": Path("./outputs/lowdim/2025-10-20/MINE-DV"),
            "InfoNCE": Path("./outputs/lowdim/2025-10-20/InfoNCE"),
        },
        "target": {
            "function": lambda x : (x["distribution.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Ground-truth Mutual Information",
        },
        "column_to_chart": {
            "name": "distribution.mutual_information",
            "apply": int,
        },
        "rows_to_chart": [
            "Distribution",
            "Estimator",
        ],
        "fixed_columns": {"n_samples": 1000},
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
            "MINE-DV": {
                "min_columns": ["estimator.estimator.backbone_factory.hidden_dim"],
                "fixed_columns": {"estimator.estimator.estimate_fraction": 0.5},
                "priority": 1,
            },
            # "InfoNCE": {
            #     "min_columns": ["estimator.estimator.backbone_factory.hidden_dim"],
            #     "fixed_columns": {"estimator.estimator.estimate_fraction": 0.5},
            #     "priority": 1,
            # },
            # "MINDE-c": {
            #     "min_columns": ["estimator_arch","mi_sigma"],
            #     "fixed_columns": {"importance_sampling": True, "estimator_type": "c"},
            #     "priority": 2,
            # },
            # "MINDE-j": {
            #     "min_columns": ["estimator_arch","mi_sigma"],
            #     "fixed_columns": {"importance_sampling": True, "estimator_type": "j"},
            #     "priority": 2,
            # },
        },
    },
    
    "mi_estimate:base:distribution_method:GT:10k": {
        "data_paths": {
            "KSG":     Path("./outputs/lowdim/2025-10-20/KSG"),
            "WKL":     Path("./outputs/lowdim/2025-10-20/WKL"),
            "MINE-DV": Path("./outputs/lowdim/2025-10-20/MINE-DV"),
            "InfoNCE": Path("./outputs/lowdim/2025-10-20/InfoNCE"),
        },
        "target": {
            "function": lambda x : (x["distribution.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Ground-truth Mutual Information",
        },
        "column_to_chart": {
            "name": "distribution.mutual_information",
            "apply": int,
        },
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
            "MINE-DV": {
                "min_columns": ["estimator.estimator.backbone_factory.hidden_dim"],
                "fixed_columns": {"estimator.estimator.estimate_fraction": 0.5},
                "priority": 1,
            },
            # "InfoNCE": {
            #     "min_columns": ["estimator.estimator.backbone_factory.hidden_dim"],
            #     "fixed_columns": {"estimator.estimator.estimate_fraction": 0.5},
            #     "priority": 1,
            # },
            # "MINDE-c": {
            #     "min_columns": ["estimator_arch","mi_sigma"],
            #     "fixed_columns": {"importance_sampling": True, "estimator_type": "c"},
            #     "priority": 2,
            # },
            # "MINDE-j": {
            #     "min_columns": ["estimator_arch","mi_sigma"],
            #     "fixed_columns": {"importance_sampling": True, "estimator_type": "j"},
            #     "priority": 2,
            # },
        },
    },

    "mixing_labels": {
        "data_paths": {
            "MINE-DV":     Path("./outputs/mixing/2025-12-14/Conv2d-MINE"),
            "MINE-NWJ":    Path("./outputs/mixing/2025-12-14/Conv2d-NWJ"),
        },
        "target": {
            "function": lambda x : (x["processed.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Ground-truth Mutual Information",
        },
        "column_to_chart": {
            "name": "processed.mutual_information",
            "apply": lambda x : f"{x:.1f}",
        },
        "rows_to_chart": [
            "Estimator",
        ],
        "fixed_columns": {"name.distribution": "labels/MNIST"},
        "estimators": {
            "MINE-DV": {
                "min_columns": ["estimator.backbone_factory.hidden_dim", "estimator.backbone_factory.n_filters"],
                "fixed_columns": dict(), #{"estimator.estimate_fraction": 0.5},
                "priority": 0,
            },
            "MINE-NWJ": {
                "min_columns": ["estimator.backbone_factory.hidden_dim", "estimator.backbone_factory.n_filters"],
                "fixed_columns": dict(), #{"estimator.estimate_fraction": 0.5},
                "priority": 0,
            },
        },
    },

    "mixing_multiplication": {
        "data_paths": {
            "MINE-DV":     Path("./outputs/mixing/2025-12-14/Conv2d-MINE"),
            "MINE-NWJ":    Path("./outputs/mixing/2025-12-14/Conv2d-NWJ"),
        },
        "target": {
            "function": lambda x : (x["processed.mutual_information"] - x["mutual_information.mean"]).abs(),
            "name": "Ground-truth Mutual Information",
        },
        "column_to_chart": {
            "name": "processed.mutual_information",
            "apply": lambda x : f"{x:.1f}",
        },
        "rows_to_chart": [
            "Estimator",
        ],
        "fixed_columns": {"name.distribution": "modulation/multiplication/MNIST"},
        "estimators": {
            "MINE-DV": {
                "min_columns": ["estimator.backbone_factory.hidden_dim", "estimator.backbone_factory.n_filters"],
                "fixed_columns": dict(), #{"estimator.estimate_fraction": 0.5},
                "priority": 0,
            },
            "MINE-NWJ": {
                "min_columns": ["estimator.backbone_factory.hidden_dim", "estimator.backbone_factory.n_filters"],
                "fixed_columns": dict(), #{"estimator.estimate_fraction": 0.5},
                "priority": 0,
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


if __name__ == "__main__":
    #data = bebeziana.read(data_path, ["setup.yaml", "results.yaml"])

    for table_name, table_config in tables.items():
        print(table_name)
        
        final_data = []
        for estimator_name, estimator_config in table_config["estimators"].items():
            data_path = table_config["data_paths"][estimator_name] / "data.csv"
            if data_path.exists():
                data = pandas.read_csv(data_path)
            else:
                data = bebeziana.read(table_config["data_paths"][estimator_name], ["setup.yaml", "results.yaml"])
                data.to_csv(data_path)

            for column, value in table_config["fixed_columns"].items():
                data = data[data[column] == value]

            for column, value in estimator_config["fixed_columns"].items():
                data = data[data[column] == value]

            data["Estimator"] = f"_{estimator_config['priority']}_{estimator_name}"
            data["Distribution"] = data["name.distribution"].map(distribution_names).fillna(data["distribution._target_"])

            data[table_config["target"]["name"]] = table_config["target"]["function"](data)
            data[table_config["column_to_chart"]["name"]] = data[table_config["column_to_chart"]["name"]].apply(
                table_config["column_to_chart"]["apply"]
            )

            data = pandas.DataFrame(
                data.groupby(
                    table_config["rows_to_chart"] + [table_config["column_to_chart"]["name"]] + estimator_config["min_columns"],
                    dropna=False
                )[[table_config["target"]["name"]]].mean()
            )

            data = pandas.DataFrame(
                data.groupby(
                    table_config["rows_to_chart"] + [table_config["column_to_chart"]["name"]],
                    dropna=False
                )[[table_config["target"]["name"]]].min()
            )

            data = data.unstack(table_config["column_to_chart"]["name"])

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

        with open(Path("./tables") / table_name, 'w') as table_file:
            table_file.write(table)