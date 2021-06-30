# Sequential Learning App
Here we present an app for accelerating the experimental search for suitable materials. It can be used for method development and for investigating the configuration of Sequential Learning (SL) methods. 
To determine the performance of SL methods, it is common to use simulated experiments where the ground truth labels for all data points are already known. Initially, only a small fraction is provided to the SL algorithm (although more training data would be available). This is extended with one new data point from the remainder of the available data at each iteration. It is investigated which approach requires the least amount of data to achieve the goal. Approaches that require less data simply lead to faster success in laboratory practice. Thus, the goal is not to actually discover new materials using all available data, but to validate material discovery methods for scenarios where fewer labels are known (e.g., for new materials).
The app provides flexible and low-threshold access to AI methods via user interfaces. It is based on "Jupyter Notebooks" and integrates seamlessly with the "AIIDA" workflow environment. The underlying code can be easily customized and extended. The app has intuitive and interactive user interfaces for data import and cleansing/selection, (statistical) data analysis, visualization for exploration and plausibility, AI environment as well as data evaluation and result visualization. Structured material data from CSV files are used.

## Installation

When not hosted on Aiidalab Plattform, this App needs to install some requirements.


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.txt file.
Navigate to the Directory where you downloaded this Repository and execute this command in your Terminal.

### Python 3
```bash
pip3 install -r requirements.txt
```
### Python 2

```bash
pip install -r requirements.txt
```
After installing the requirements enable the ui elements

```bash
jupyter nbextension enable --py widgetsnbextension
```

## Usage

Navigate to the Directory where you downloaded this Repository and execute this Command in your Terminal

```bash
voila SequentialLearningApp.ipynb
```

A Window in your default browser should open now.

## Background on the used methodology: 

More information on the methodology is available in the publication entiteled "Sequential learning to accelerate materials discovery of alkali-activated binders" ( DOI: 10.13140/RG.2.2.18388.94087 ). 

# Hands on - a quick guide to the sequential learning app

The app is divided into the four main windows "Upload", "Data Info", "Design Space Explorer" and "Sequenital Learning", which are explained below. 

## Upload
In the upload window of the app, the material data in CSV format can be imported via a dialog. An example file is provided in this repository. An option to set the CSV separator, the decimal separator and to delete non-numeric data is available. Additionally, lines at the beginning of the file can be skipped (e.g., header information, etc.). At the end of this process, the data is displayed to the user to allow a plausibility check. Here it can be checked quickly and easily whether decimal places are specified correctly, and all data are numeric.

## Data Info
This window gives a detailed overview of the uploaded data. Besides the data preview, there is a detailed list of all variables (Info button) and some basic statistical characteristics of the variables (Stats button).

## Design Space Explorer
The Design Space Explorer allows the visualization of complex relationships in the data. Here, specific dependencies between selected variables can be displayed as a scatter plot, the interrelationships and distributions of the variables can be mapped as a scatter matrix, and correlations can be visualized as a correlation heatmap. These tools allow a quick visual overview, e.g. of co-linearities of the characteristics for feature selection or trade-offs between different material properties, which are to be optimized.

## Sequential Learning
This window provides a SL framework divided into the tabs "Settings" - here the optimization scenario can be defined - and "Seqential Learning Parameters" - here the algorithms can be selected, set and virtual experiments can be performed. 

### Settings
This window lets the user interactively set up the boundary conditions of the SL problem. The input feature and target properties can be selected simply by mouse click. It is possible to select multiple target properties (Multi-Objective Optimization). The optimization is then based on the sum (or difference - depending on whether maximization or minimization is desired) of the normalized properties. 
The target can be specified as a quantile of the given properties (or their combinations in case of Multi-Objective Optimization). A lower target threshold (e.g. 90%) accelerates the SL optimization. However, this makes it increasingly difficult for SL to outperform a random process. 
The sample threshold determines the restriction of the initial training data. If it was not restricted, it would be possible that the searched material is already contained in the initial data - which would make SL superfluous. The sample threshold is therefore always lower than the target threshold. 
The initial sample size can be chosen below. Some SL algorithms require at least 3 samples. It is recommended to not choose less than 4 initial samples. 

### Sequential Learning Parameters
This tab lets the user select from several Machine Learning algorithms and utility functions. Some utility functions, such as MEID and MLID, allow to adjust further hyper parameters. The number of randomized SL runs can be set with the “# of SL runs” slider (standard value=30). The “Run” button executes a simulated experiments where the selected SL algorithms solve the optimization problem that has been specified in the “Settings” tab for the set number of SL runs. 

#### Result diagrams
The first diagram shows how fast a selected SL algorithm can find its way to the target. This is shown for each SL run as a linineplot in terms of the minimum distances in the design space from the already discovered materials to the targets per SL iterations.  If the discovered materials remain far from the target solution for many iterations, a more explorative approach may help to improve performance. If it converges quickly, a more exploiting algorithm may improve performance even further. 
The histogram below compares the performance in terms of experiments required of the SL algorithm VS a random process. SL is typically compared to a random process (RP) (i.e., without strategy or model) as a baseline benchmark. RPs consider each candidate as equally likely to succeed (uniform distribution). However, the success rate of RP has a nonlinear relationship with the required draws for the case of multiple targets, (The set of targets is controlled by the target threshold). A low target threshold means that RP becomes a much more difficult benchmark.

## Conclusion
Serial data collection of SL, even if more successful than RP, can be detrimental in a real-world application, as waiting for experimental results could delay experimental progress. This is especially the case for materials whose synthesis is complex and whose material properties take time to develop or characterize (e.g., 28-day compressive strength of concrete). Collecting all samples at once or in batches may be more successful. 
SL therefore targets material innovations for which data are not available or large-scale data collection would be too expensive. However, what actually provides an advantage in the lab depends heavily on how the SL problem is designed. The purpose of this app is to provide a tool to explore exactly under what conditions SL can help accelerate research. 



