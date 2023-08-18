# Stock Trend Analyzer

## Note:
You are on the wrong branch, this is where I actively make changes here and a number of things probably do not work. Go back to the main branch and clone that one.

## Table of Contents

- [Stock Trend Analyzer](#stock-trend-analyzer)
  - [Note:](#note)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Usage ](#usage-)

## About <a name = "about"></a>

Using XGDBoost And A custom neural Network to analyse the gradients of stocks. 

I highly recommend creating a new environment for every project.This will create the env and activate it

```
python3 -m .venv venv
source venv/bin/activate
```

This will install all the needed deps
```
pip install -r requirements.txt
```

## Usage <a name = "usage"></a>

- `notebooks/model.ipynb` : Consists a d>iverse set of technical indicators that it evaluates off of, input can be any notebook ending with `usable.ipynb`.
- `notebooks/model2.ipynb` : Consists a less diverse set of techincal indicators however, gives a higher accuracy. Both these notebooks are based on xgboost's regression model
- `notebooks/nueral_net.ipynb` : A custom neural network written in pytorch, takes the longest to run (~1min), however does provide the the most accuracy. Input can be any data directly from the NSE webitse or directly from yahoo finance.
- `docs/neural_network/` : consists of all the notebooks run on the neural network along with their accuracy and some test predictions
- `docs/regressor/`: consists of all the notebooks run on the xgboost regressor model along with their accuracy and predictions 