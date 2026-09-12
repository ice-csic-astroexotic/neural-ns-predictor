# neural-ns-predictor

## Installation instructions

### Step 1: get the code and data

Download or clone the repository, and enter the main folder ```neural-ns-predictor```.

### Step 2: seeting up the conda environment

Create a clean conda environment using the following command:

```bash
conda create -n neuralns python=3.11 -y
```

and then activate it:

```bash
conda activate neuralns
```

Optionally, you can use a different name by changing `neuralns`.

### Step 3: installing the package

#### Linux

From the top folder, run the following command to install the package:
```pip install neural-ns-predictor```
Developers that modify the package, can install it as an editable package
(to avoid re-install every time) in the following way:
```pip install -e neural-ns-predictor```

#### Mac OS

From the top folder, run the following command to install the package:
```python -m pip install .```
Developers that modify the package, can install it as an editable package
(to avoid re-install every time) in the following way:
```python -m pip install -e .```

If there are problems with the rendering of LaTeX equations in the plots created in the notebooks, please run:
```brew install --cask mactex-no-gui```

## Application 1: The catalog paper

The provided scripts (from 1 to 6) need a specific folder structure to work:

A `data` folder providing the pulsar properties, and a `results` folder where
the results will be saved in CSV format. Both folders should be located two
levels above the scripts. For example:

```ascii
      data
      results
      neural-ns-predictor
          ...
          scripts
              ...
              6_catalog_application.ipynb
```

By running the 6th script (see above) one can derive the results published in
the paper.
