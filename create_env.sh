conda create -y -n GatedGPS python=3.11

source activate GatedGPS

conda install -y cudatoolkit=11.8

conda install mkl=2023.1.0 -y # install if missing in torch environment

conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip uninstall numpy -y

pip install numpy==1.26.4

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install wandb

pip install fair-esm

pip install timm scikit-learn