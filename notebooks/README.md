# Experiment Notebooks

Neuro-nav includes a number of interactive jupyter notebooks. These demonstrate features of the library, as well as serve to reproduce various experiments in the literature.

## General Notebooks

| Notebook | Description | Referenced Work | Animal Model | Colab Link |
| ------------- | ----------- | ---------- | --------------- | --- |
| [Usage Tutorial](./usage_tutorial.ipynb) | This notebook provides a basic usage tutorial of both environment types. This is the best place to start for those seeking to understand the features of neuro-nav. | N/A | N/A | [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/usage_tutorial.ipynb) |
| [CCN 2022 Tutorial](./ccn_tutorial.ipynb) | This notebook was made to accompany the CCN 2022 Tutorial "Varieties of Human-like AI." It provides an overview of a number of the algorithms included in Neuro-Nav, and their properties. | N/A | N/A | [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/ccn_tutorial.ipynb) |


## Cognitive Neuroscience Notebooks

| Notebook | Description | Referenced Work | Animal Model | Colab Link |
| ------------- | ----------- | ---------- | --------------- | --- |
| [Successor Representation Experiments](./representation_experiments.ipynb) | Demonstrates how to generate visualizations of the learned representations from agents utilizing a successor representation. These include value maps, successor "place" cells, and successor "grid" cells. | [Stachenfeld et al., 2017](https://www.nature.com/articles/nn.4650) | Rodent |  [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/representation_experiments.ipynb) |
| [Grid Transfer Experiments](./grid_experiments.ipynb) | Evaluates various algorithms ability to adapt to changes in reward location or environment structure in goal-directed navigation tasks. | [Russek et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) | Rodent | [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/grid_experiments.ipynb) |
| Graph Transfer Experiments [(1)](./graph_experiments_1.ipynb) [(2)](./graph_experiments_2.ipynb) | Evaluates various algorithms ability to adapt to changes in reward contingencies or transition dynamics in decision making tasks. | [Momennejad et al., 2017](https://www.nature.com/articles/s41562-017-0180-8) | Human |  [Link (1)](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/graph_experiments_1.ipynb), [Link (2)](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/graph_experiments_2.ipynb) |
| [Temporal Community Experiments](./temporal_community.ipynb) | Compares methods for learning representations which display temporal community structure. Utilizes a graph environment with local neighborhood structure. | [Schapiro et al., 2013](https://www.nature.com/articles/nn.3331), [Stachenfeld et al., 2017](https://www.nature.com/articles/nn.4650) | Human | [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/temporal_community.ipynb) |
| [Distributional Value Experiments](./distributional.ipynb) | Compares a distributional and classical TD algorithm on a variable reward magnitude task. The distributional TD algorithm better captures behavior of dopamine neurons. | [Dabney et al., 2020](https://www.nature.com/articles/s41586-019-1924-6) | Rodent | [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/distributional.ipynb) |

## Computational Psychiatry Notebooks

| Notebook | Description | Referenced Work | Animal Model | Colab Link |
| ------------- | ----------- | ---------- | --------------- | --- |
| [Optimism & Pessimism Experiments](./pessimism_experiments.ipynb) | Demonstrates maladaptive learning when pessimistic value bootstrapping is used for distal state updates. | [Zorowitz et al., 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8143038/) | Human | [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/pessimism_experiments.ipynb) |
| [Mood Experiments](./mood_experiments.ipynb) | Demonstrates learning mood as the temporal integral of advantages, both in its causes and effects. | [Eldar et al., 2016](https://www.sciencedirect.com/science/article/pii/S1364661315001746), [Bennett et al., 2022](https://psycnet.apa.org/record/2021-84803-001) | Human | [Link](https://colab.research.google.com/github/awjuliani/neuro-nav/blob/main/notebooks/mood_experiments.ipynb) |
