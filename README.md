# Stock Trend Analyzer

## Table of Contents

- [Stock Trend Analyzer](#stock-trend-analyzer)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
  - [Usage ](#usage-)

## About <a name = "about"></a>

Using XGDBoost to analyse the gradients of stocks, using a regressive algorithm with weights depending  on the data

## Getting Started <a name = "getting_started"></a>

Firstly, you need to have a setup for notebooks, either on Jupyter/VSCode/Neovim,etc. To install all of the dependencies, first create a new environment and install the deps.

This will create the env

```
python3 -m .venv venv
```

This will install all the needed deps
```
pip install -r requirements.txt
```

## Usage <a name = "usage"></a>

There are 4 main notebooks (for now). This `notebooks/weekly_data_creation.ipynb` is responsible for creating and formatting the raw data. The notebook assumes that the at this point in the process, your notebook may not have any technical indicators. The output of this notebook must be fed into this `notebooks/add_technical_indicators.ipynb` notebook. The function of this notebook is to add 7 technical indicators (I highly suggest messing around with these to get the best personal output). I have the left the module in a state where I was getting the most optimal output.
The output of this notebook must be ran thorugh the `notebooks/model.ipynb`. This notebook will create the model and perform inference as well. 

This project is under development. I will add a neural network at some point to continue this.