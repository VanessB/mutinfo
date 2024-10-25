import bebeziana
import yaml

from pathlib import Path


if __name__ == "__main__":
    with open("./config.d/config.yaml", 'r') as file:
        config = yaml.load(file, yaml.Loader)

    data_path = Path("./outputs/2024-10-25/")
    data = bebeziana.read(data_path, ["setup.yaml", "results.yaml"])

    data.to_csv(data_path / "data.csv")
