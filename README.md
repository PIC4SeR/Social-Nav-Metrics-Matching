[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](https://arxiv.org/abs/2107.00606)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 


<h1 align="center">  Social Navigation Metrics Matching
</h1>

<!-- [Graphical abstract goes here]
<p align="center">
  <img src="https://amlbrown.com/wp-content/uploads/2015/10/11219225_10153619513398446_2657606012680909527_n.jpg" alt="Alternative text" width="450"/>
</p> -->

## Objective of the project

The goal of the project is to analyze the correlation between existing social navigation metrics commonly used in robot evaluation and metrics assigned by humans through surveys. The project aims at highlighting key insights that emerge from such comparison: are current metrics exaustive to represent the navigation experiment compared to human-level opinion? What aspect is missing? 

**Definiton**

QM := Quantitative Metrics

HM := Human Metrics

## Roadmap

- **Goal 0**:
   - Setting up data structure to import evaluation metrics data in np arrays (done)
   - Labels assigned to each experiment according to the assumptions (done)
   - Plotting utils to start showing bar plots, also with standard deviation (in progress)

- **Goal 1** (Work in Progress): Investigate overall correlation between quantitative and qualitative metrics. 
  - 1.1: Discretize quantitative metrics in 1-5 score range and plot the histogram for each experiment to compare with qualitative results
  - 1.2: Cluster the experiments in the 3 expected clusters (based on assigned labels) and compare the results obtained using QM space and HM space

- **Goal 2** (not started): Single metrics feature matching. Which aspect of human evaluation is missing in the existing quantitative metrics? 
  - **Output**: table of regression score matching
  - 2.1. ANOVA (variance analysis). Helps to verify whether the differences in the metric produce significant differences in the rating.
	- 2.2. Analysis of correlation between each metric and rating (average values) (Spearman and/or Kendall).
	- 2.3. Optionally, to model the relation between metric and rating, ordinal regression

- **Goal 3**(not started): Which are better to classify correctly the experiment? 
Train a classifier to predict the goodness of the experiment based on assigned labels using both QM and HM features. 
Decision trees and many others, and applying explanability methods to know the "relevance" of each metric

## Current state of works
- Goal 0 (done): Setting up data structure to import evaluation metrics data in np arrays
- Goal 1 (quite done): Investigate overall correlation between quantitative and qualitative metrics. We set up plots for direct comparison of average results from survey and quantitative lab data. Then, we use K-means clustering to compare the category assigned to each epxeriment based on QM features os Survey HM features.

## Installation [TODO]
[Explanation on how to install the project goes here]

Always use ```pipreqs``` to generate the requirements.txt file.
```
pip install -r requirements.txt
```
Don't forget to provide links to datasets, model weights, etc.

## Usage
[Explanation of basic usage goes here]

Provide sample scripts if needed.

```
python main.py --mode train --gpu 0
```

# Citations
Remind users to cite your work, e.g.:

This repository is intended for scientific research purposes.
If you want to use this code for your research, please cite our work ([Paper Name](https://arxiv.org/)).

```
[.bib citation here]
```

# References
[Other references that should be cited when using this repository here]

# Acknowledgements
[Acknowledgements here]

<p align="left">
  <img src="https://media.giphy.com/media/yWh7b6fWA5rJm/giphy.gif?cid=790b7611ieiiqtp06t9x5bju00gzcgryrw8me999ep27ovcj&ep=v1_gifs_search&rid=giphy.gif&ct=g" alt="animated" />
</p>
