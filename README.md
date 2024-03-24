# Custom Python Functions

This repository is a collection of custom functions for data science and machine learning.

## Model_evaluation.py

This file include a set of functions for training, cross-validation, prediction, and valuation of machine learning models. 

To use these functions, add the following lines to the beginning of the python notebook where the import command lines are:

```python
from os import path
import sys
sys.path.append(path.abspath('../')) #replace this path with your own path
from custom_python_functions.model_evaluation import (
    train_crossval_predict_score,
    predict_and_print_scores,
    plot_confusion_matrix,
    plot_distributions,
    plot_correlation,
    plot_roc_curves)
```

## Useful_syntax.ipynb

This notebook includes a series of python code usefull to perform data science tasks.
