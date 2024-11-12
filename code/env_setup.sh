#!/bin/bash
# Install torch+cuda
export CUDA_HOME=/usr/local/cuda # depends on your CUDA configuraiton
pip3 install Ninja packaging

python -m pip install Cython


python setup.py build_ext --inplace
# python setup.py install
pip3 install --editable .

## install ctcdecode
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ..

## Install apex
# pip3 install Ninja
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

cd dag_search
bash install.sh
cd ..

## Others
pip3 install hydra-core==1.1.1

pip3 install sacremoses
pip3 install 'fuzzywuzzy[speedup]'
pip3 install git+https://github.com/dugu9sword/lunanlp.git
pip3 install omegaconf
pip3 install nltk
pip3 install sacrebleu==1.5.1
pip3 install sacrebleu[ja]
pip3 install scikit-learn scipy
pip3 install bitarray
pip3 install tensorboardX
pip3 install tensorflow # ==2.3
# pip install git+https://github.com/chenyangh/sacrebleu.git@1.5.1

pip install gdown
pip3 install scipy
pip3 install wandb
pip install rouge
pip install rouge_score

# pip install streamlit==0.62.0
# pip install pyarrow==0.12.0
# pip install --extra-index-url https://pypi.fury.io/arrow-nightlies/ --prefer-binary --pre pyarrow


pip install git+https://github.com/tagucci/pythonrouge.git
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install transformers
pip install evaluate # pip install --no-deps evaluate
pip install bert-score
conda upgrade numpy

wandb login 6dc39255e119c5d8d2e39a809e9edfe896c3633e

# module load boost
# module load cmake
