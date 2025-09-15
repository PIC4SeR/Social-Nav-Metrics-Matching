[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](https://arxiv.org/abs/2107.00606)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 


<h1 align="center">  Metrics vs Surveys: Can Quantitative Measures Replace Human Surveys in Social Robot Navigation? A Correlation Analysis
</h1>

<!-- [Graphical abstract goes here]
<p align="center">
  <img src="https://amlbrown.com/wp-content/uploads/2015/10/11219225_10153619513398446_2657606012680909527_n.jpg" alt="Alternative text" width="450"/>
</p> -->
<p align="center">
  <img src="https://anonymous.4open.science/r/Social-Nav-Metrics-Matching-ED7D/images/Corr-Metrics.drawio.png" alt="Correlation Metrics Analysis" width="450"/>
</p>

## Objective of the project


Social (also called human-aware) navigation is a
key challenge for the integration of mobile robots into human
environments. The evaluation of such systems is complex, as
factors such as comfort, safety, and legibility must be consid-
ered. Human-centered assessments, typically conducted through
surveys, provide reliable insights but are costly, resource-
intensive, and difficult to reproduce or compare across systems.
Alternatively, numerical social navigation metrics are easy to
compute and facilitate comparisons, yet the community lacks
consensus on a standard set of metrics.
This work explores the relationship between numerical
metrics and human-centered evaluations to identify potential
correlations. If specific quantitative measures align with human
perceptions, they could serve as standardized evaluation tools,
reducing the dependency on surveys. Our results indicate that
while current metrics capture some aspects of robot navigation
behavior, important subjective factors remain insufficiently
represented and new metrics are necessary.

This repository contains the collected metrics for the 24 analyzed experiments, along with the necessary files used to analyze them.

8 scenarios were tested, with 3 different controller configurations each to have a different behavior, leading to diverse runs as can be seen on the gifs below
<p align="left">
  <img src="https://anonymous.4open.science/r/Social-Nav-Metrics-Matching-ED7D/images/first_passing.gif" alt="First Passing" width="300" style="display: inline-block; margin-right: 20px;"/>
  <img src="https://anonymous.4open.science/r/Social-Nav-Metrics-Matching-ED7D/images/second_passing.gif" alt="Second Passing" width="300" style="display: inline-block; margin-right: 20px"/>
  <img src="https://anonymous.4open.science/r/Social-Nav-Metrics-Matching-ED7D/images/third_passing.gif" alt="Third Passing" width="300"
  style="display: inline-block;"/>
</p>

The human metrics were collected via survey, survey that can be viewed at this [link](https://docs.google.com/forms/d/e/1FAIpQLSf_Dl9Fxwj-b9akZzA06BRVu8GOQalZP8z9UsHuONQnMncChA/viewform?usp=dialog). This is an exact copy of the original survey; the original isn't shown to avoid tampering with the answers given.

##


**QM := Quantitative Metrics, 11 in total**

**QM Metrics**
- [0] Time to Goal
- [1] Path length
- [2] Cumulative heading changes
- [3] Avg robot linear speed

- [4] Social Work 
- [5] Social Work (per second)
- [6] Average minimum distance to closest person
- [7] Proxemics: intimate space occupancy
- [8] Proxemics: personal space occupancy
- [9] Proxemics: social space occupancy
- [10] Proxemics: public space occupancy

**HM := Human Metrics, 4 in total**

**HM Metrics**
- [0] Unobtrusiveness
- [1] Friendliness
- [2] Smoothness
- [3] Avoidance Foresight

**ARI := Adjusted Rand Index**



## Installation

Odfpy is required to extract data from the ods and Excel files
```
pip install odfpy
```
<!--Always use ```pipreqs``` to generate the requirements.txt file.
```
pip install -r requirements.txt
```-->
The metrics are already extracted and presented in the data_folder, the recorded ros2 bags associated with them can be downloaded at this [link](https://drive.google.com/file/d/1DMiw7qAvpCDC3eAf4Af-XRlvGY9o6DQf/view?usp=drive_link)



## Usage
The analysis is carried out in the three included Jupyter notebook files.
The home path in the three notebooks as to be changed to the relative path of the repository.

**G1_1_overall_metrics_match.ipynb**

This notebook shows the results presented in the "Overall evaluation comparison" section, thanks to the reduced set of QM metrics (0,3,6,7,9). The functions for plotting the histograms are presented.

**G1_2_cluster_social_metrics.ipynb**

This notebook shows the results presented in the  "Clustering comparison" section. The ARI is computed to identify the best combinations for clustering, and the cumulative ARI is then calculated to determine the most impactful metrics, which are plotted in a histogram.

**G2_statistical_analysis.ipynb**

This notebook shows the results presented in the "Statistical analysis for metrics correlation" section. The Spearman and Kendall coefficients are computed and the data is presented in a heatmap to show the most relevant metrics.




<!--# Citations
Remind users to cite your work, e.g.:

This repository is intended for scientific research purposes.
If you want to use this code for your research, please cite our work ([Paper Name](https://arxiv.org/)).

```
[.bib citation here]
```

<!--# References
[Other references that should be cited when using this repository here]

# Acknowledgements
[Acknowledgements here]
-->
