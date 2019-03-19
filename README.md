# Toxic-Comments-Classification
This repository contains the python code created with pytorch to solve the problem of toxic comments classification in social media.
Kaggle competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

--------------------

## Data Overview:

- data/ : contains the kaggle datasets
- saved_models/ : contains saved neural networks
- loader.py : handles all data loading for training
- models.py : contains the definition of the neural networks
- train.py
- util.py
- visualise_data.py : some function for data visualisation

--------------------

## Usage:

See toxic_comments.ipynb for details and usage example

--------------------

## Visualization

We're using a modified version of the data visualization code from https://github.com/abisee/attn_vis.
To visualize the attention distribution on a comment, one should run the function visualise_data.gen_visualisation(...)
and then run a

```
cd attention_vis/
python -m http.server
```

and go to the index page.