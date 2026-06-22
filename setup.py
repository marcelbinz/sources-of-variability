import tarfile
import os
import subprocess

os.makedirs("data", exist_ok=True)

with tarfile.open("data/SharedResponses.csv.tar.gz", "r:gz") as tar:
    tar.extractall(path="data", filter="data")

# Create conda environment
subprocess.run(["conda", "env", "create", "-f", "SoV.yaml"], check=True)
