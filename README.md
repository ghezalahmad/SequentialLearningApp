# Sequential Learning App
Concrete Strength Prediction with Sequential Learning

Here we present an app for accelerating the experimental search for suitable materials. It can be used for method development and for investigating the configuration of Sequential Learning (SL) methods. The app provides flexible and low-threshold access to AI methods via user interfaces. 
The app, based on "Jupyter Notebooks", integrates seamlessly with the "AIIDA" workflow environment. The underlying code can be easily customized and extended. The app has intuitive and interactive user interfaces for data import and cleansing/selection, (statistical) data analysis, visualization for exploration and plausibility, AI environment as well as data evaluation and result visualization. Structured material data from CSV files are used.

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

A Window in your default browser should open now. The rest is pretty self explaining.

## Background

SL and the closely related Bayesian optimization have repeatedly been reported to have great potential in accelerating drug and material discovery. The basic idea is to reduce the number of unsuccessful experiments (i.e., that lead to materials with unwanted properties) so that an ideal sequence of successive experiments is achieved. This is accomplished by coupling a prediction model (e.g., a machine learning model) with a decision-making rule based on a so-called utility function that guides the experimental program.
