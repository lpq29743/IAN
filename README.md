# IAN

A Tensorflow implementation for ["Interactive Attention Networks for Aspect-Level Sentiment Classification"](http://static.ijcai.org/proceedings-2017/0568.pdf) (Dehong Ma, IJCAI 2017)

## Quick Start

- Create three empty folders: 'analysis' for saving analyzing results, 'logs' for saving experiment logs and 'models' for saving experiment models 
- Download the 300-dimensional pre-trained word vectors from [Glove](https://nlp.stanford.edu/projects/glove/) and save it in the 'data' folder as 'data/glove.6B.300d.txt'

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
| Laptop     | 69.436   |
| Restaurant | 78.571   |

