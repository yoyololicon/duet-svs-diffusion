# LASS

## Installation
Installing the necessary dependencies can be done through conda, by executing the following commands:
```bash
cd lass_mnist
conda env create -f environment.yaml
conda activate lass_mnist
``` 
Once the conda environment is installed it is possible to start the separation procedure.

## Download Pre-trained models
You can download the necessary checkpoints from [here](https://drive.google.com/file/d/1oayY1FEUrTwQJMr78mP1t6r8AggjzAso/view?usp=share_link). 
Place the downloaded file inside of the directory `lass_mnist/checkpoints` and extract it with the command `tar -xf lass-mnist-ckpts.tar`.

## Separate images
To separate MNIST test, you can run the script
```bash
PYTONPATH=. python lass/separate.py 
``` 
This will separate the test set of mnist into the folder `separated_mnist`.
