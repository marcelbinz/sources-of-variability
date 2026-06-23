import tarfile
import os
import subprocess

os.makedirs("data", exist_ok=True)

with tarfile.open("data/SharedResponses.csv.tar.gz", "r:gz") as tar:
    tar.extractall(path="data", filter="data")


# install mamba for memory efficiency
subprocess.run(["conda", "install", "-n", "base", "-c", "conda-forge", "mamba"], check=True)
# Create conda environment
subprocess.run(["mamba", "env", "create", "-f", "SoV.yaml"], check=True)
