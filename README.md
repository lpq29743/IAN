# IAN

A Tensorflow implementation for ["Interactive Attention Networks for Aspect-Level Sentiment Classification"](https://arxiv.org/pdf/1709.00893.pdf) (Dehong Ma, IJCAI 2017)

## Quick Start

- use `pip install -r requirements.txt` to install required packages
- Create three empty folders: 'analysis' for saving analyzing results, 'logs' for saving experiment logs and 'models' for saving experiment models 
- Download the 300-dimensional pre-trained word vectors from [Glove](https://nlp.stanford.edu/projects/glove/) and save it in the 'data' folder as 'data/glove.840B.300d.txt'

## Source Code Tree

```
|--- data

|	|--- laptop

|	|--- restaurant

|	|--- data_info.txt - the preprocessing data information file

|	|--- test_data.txt - the preprocessing testing data file

|	|--- train_data.txt - the preprocessing training data file

|--- main.py

|--- model.py

|--- transfer.py - transfering the origin xml files to text files

|--- utils.py

|--- README.md
```

## Results

| Dataset    | Accuracy |
| ---------- | -------- |
| Laptop     | 70.846   |
| Restaurant | 79.107   |

Note: In the newest version, the results are worse than the results given above, since the code of the model is revised.
I will optimize the model sooner and report the results.

## Todo List

- Implementing by other deep learning frameworks
- Softmax mask
- Optimization to get better performance
