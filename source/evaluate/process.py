import bebeziana
import yaml

from pathlib import Path


if __name__ == "__main__":
    folder_name = input("Enter the folder name: ")
    
    with open("./config.d/config.yaml", 'r') as file:
        config = yaml.load(file, yaml.Loader)

    data_path = Path(f"{folder_name}")
    data = bebeziana.read(data_path, ["setup.yaml", "results.yaml"])

    data.to_csv(data_path / "data.csv")
