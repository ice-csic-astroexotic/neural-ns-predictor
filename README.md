# neural-ns-predictor
# Installation instructions
## Step 1: get the code and data
Download or clone the repository, and enter the main folder ```neural-ns-predictor```.

## Step 2: seeting up the conda environment
Create a clean conda environment using the following command:
```bash
conda env create -f environment.yml -n neuralns
```
and then activate it:
```bash
conda activate neuralns
```
Optionally, you can use a different name by changing `neuralns`.

## Step 3: installing the package
From the top folder, run the following command to install the package:
```pip install .```
Developers that modify the package, can install it as editable in the following way:
```pip install -e .```
# Application 1: The catalog paper
