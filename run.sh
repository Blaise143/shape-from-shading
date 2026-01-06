#!/usr/bin/env bash

set -e

ENV_NAME="sfs"
PYTHON_VERSION="3.11"

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -r requirements.txt

python main.py \
  --images-dir data/images \
  --masks-dir data/masks \
  --out-path out
