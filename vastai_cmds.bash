git clone https://github.com/crystaltine/position-coupling.git

curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh; chmod +x; miniconda.sh; sh miniconda.sh -b -p /content/miniconda

export PATH="/content/miniconda/bin:$PATH"

cd /workspace/position-coupling

conda env create -f env.yaml

conda init

exec bash

cd /workspace/position-coupling

conda activate NLP
