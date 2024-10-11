# SIMPLE, ACCURATE, AND EFFICIENT AXIS-ALIGNED DECISION TREE LEARNING

## Description
This repository provides implementations for the novel ProuDT experiments presented in this paper.

## Requirements
To install the dependencies, run:
```
pip install -r requirements.txt
```

<!-- or with Conda:

```
conda env create -f environment.yml
``` -->

## Experiments

Run `experiment.ipynb` to implement the ProuDT. <br>
In the paper, the experiments include 10 trials in default settings. You can specify any dataset name (see `config.py`) to reproduce the result. For example,
` 
name = "Iris"
`. All trial results and statistics will be saved in CSV files. 


## Acknowledgments

Part of the implementation for categorical dataset preprocessing is based on [GradTree](https://github.com/s-marton/GradTree). Modifications have been made to suit this project’s requirements. Special thanks to the author for their valuable contribution.









