# Sequential Learning App
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

A Window in your default browser should open now.

## Background

SL and the closely related Bayesian optimization have repeatedly been reported to have great potential in accelerating drug and material discovery. The basic idea is to reduce the number of unsuccessful experiments (i.e., that lead to materials with unwanted properties) so that an ideal sequence of successive experiments is achieved. This is accomplished by coupling a prediction model (e.g., a machine learning model) with a decision-making rule based on a so-called utility function that guides the experimental program.

The scope of this app is to provide a tool to investigate exactly under which conditions SL can contribute to accelerating research regarding the properties of Materials. For this, data from the Lab can be imported to the app. 
Based on this data, this app allows to investigate how various SL-algorithms, the quantitative research objective in terms of a target threshold, the initial sample size, and other parameters influence the performance of SL and deduce some of the critical circumstances under which SL can potentially enhance materials research practice. We introduce a novel utility function that adapts common utility functions for applications with minimal training data i.e., lower number of experiments to reach the optimal sample design. 

The underlying idea of SL is that not all experiments are equally useful. Some experiments provide more information than others. In contrast to classical design of experiments, where (only) the experimental parameters are optimized, the potential outcomes of the experiments themselves are the decisive factor. The most promising experiments are preferred over dead-end experiments and experiments whose outcome is already known. Experimental results are used to iteratively improve the ML model with high quality data. Each new experiment is selected to maximize the amount of useful information using previous experiments as a guide for the next experiment. 
The prediction of material properties in SL is based on a list of candidate materials given by experts using their domain knowledge. Materials may be of interest because they are available, cheap, known to have further desirable properties, or simply because they seem generally promising. Although the exact criteria are not specified, it is recognized that the performance of the SL for material discovery is related to the quality of the candidates. The candidate materials are represented in the so-called design space (DS) - a vector space that is comparable to the feature space in classical ML approaches. In the DS, the coordinates of each material are parameterized information about raw material, (micro-) structure and processing. An initial training data set with known target properties serves as an input for the prediction model in the first round. 
At the core of the iterative SL task is the prediction of experimental outcomes, weighting the expected utility and deciding which candidate to investigate next. The utility is commonly estimated based on the predicted material property (the closer a predicted experimental result is to the desired value, the more useful it is) and a measure of uncertainty. The latter is a key driver for discovering new relationships and the basis of experimenting in general: "Actually, the outcome of an experiment is the deviation from what we expected." In other words, if the outcome of an experiment is already known, there is no reason to conduct it and an experiment can be more useful if the uncertainty of its outcome is large. In this sense, uncertainty can be considered an essential factor in the decision-making process. The SL task is finished as soon as the desired property is obtained. 

### Prediction methods and uncertainty estimates
Originally, decision trees and tree ensembles are classification algorithms that learn the segmentation of an input data space, e.g., the DS, from pairs of data and labels [5]. By introducing one class per discrete label value (and interpolated intermediate values), pseudo-regression is performed, meaning that interpolated predictions are possible, but extrapolations outside the range of values of the label set are not. The core of tree-based algorithms is the sequential decision-making alongside the values of the respective input variables. In that sense, the data points are not considered as a "whole", but each coordinate is independently partitioned into discrete label values. By nature, this makes it relatively ill-suited to capture inter-parameter correlations. However, this can be advantageous for high-dimensional data, where unwanted correlations (so-called co-linear behavior) often result from the limited amount of available data (as expected with material data). In addition, the set of successive decisions is limited. This can result in a prioritization of relevant DS-parameters, which helps to further reduce problem complexity. 

Ensemble trees resample the training data - e.g., by a random draw with replacement - and train a new decision tree on each of the draws. The resulting ensemble tree is, depending on the respective algorithm, the average of the tree ensemble (so-called bagging) but can also have a more complex algorithmic nature that includes, for example, an error-weighted average of the trees (as in boosting). Ensemble learners generally reduce the influence of noisy training data on the prediction and create more refined decision rules. However, resampling requires slightly more data which could have a negative effect for very small data sets (as is to be expected in an early experimental stage). 
A crucial parameter of many SL methods is the uncertainty of a prediction (see section SL). More precisely, the epistemic uncertainty from the potentially erroneous assumptions of a model due to incomplete information is sought. Most ML methods do not provide an estimate of this by default because they are point estimates. However, it can be calculated as the dispersion of the prediction under slightly varying boundary conditions. To this end, varying training datasets can be created by resampling (such as jackknife bootstrapping) from the original training set. The uncertainty then corresponds to the prediction scattering of the models trained on different samples of the training. 

### Strategies and utility functions 
SL executes a strategy to select the next input by prioritizing the predictions, which are weighted by a utility function. The prioritization is conducted by – depending on whether the objective is to minimize or maximize a criterion – choosing the minimum or maximum weighted value. For simplicity, only the maximization case will be considered in the remainder of this description which can be described by the equation (1),

x_(n+1)=argmax(u)										(1)

where x_(n+1) is the selected next candidate and argmax(u) corresponds to finding the maximum utility u (note: the app also allows to minimize properties). Three general strategies can be distinguished. 1. Explorative strategies attempt to reduce model uncertainty by using utility functions that favor candidates with large prediction uncertainties. 2. Exploitative strategies tend to reinforce the current model perception by considering only the predicted values by the utility function (without considering uncertainties). 3. The third group is balancing between exploring and exploiting. Only 2. and 3. are greedy strategies and thus suitable for most material finding problems.

Maximum Expected Improvement (MEI)
The (MEI) strategy purely exploits by simply selecting the next candidate according to the maximum prediction value. The utility u_i  of the i-th prediction is simply:

u_(MEI,i)=μ_i											(2)

where μ_i is the mean prediction of the i-th candidate. 

Maximum Likelihood of Improvement (MLI)
The MLI strategy is an explore and exploit strategy. It selects the candidate with the highest likelihood to exhibit the desired target property. In the case of normally distributed prediction, the candidate with the highest 95 percent likelihood can be determined according to equation (3),

u_(MLI,i)=Q(y_i,0.95)= μ_i+1.93*σ_i								  (3)

where Q(95%) is the 95 % quantile, μ_i is the mean prediction of the i-th candidate and σ_i is the standard deviation of the i-th candidate. 

MEI and MLI with maximum Euclidean distance (MEI+D and MLI+D, respectively)
At the beginning of an SL run, the predictive power of ML algorithms is relatively poor due to the small amount of training data. The data are further reduced by sampling for uncertainty estimation by DT and TE, with only a portion of the data available for each sample. This causes a situation where many candidates yield the same prediction and uncertainty value, despite the fact that their composition and processing’s are very different. Candidates that have a large average Euclidean distance to the known DS candidates differ naturally more in their design. Their choice would increase the data variability and, in turn, the predictive model's performance will be most improved. This a-priori knowledge is for instance naturally part of "Krigging", such that it outputs higher uncertainties for more distant data points. The utility function can be adjusted in a similar way by choosing the value that has the largest mean distance to the known DS candidates from a given range of prediction values. The MEI+D or MLI+D utilities were estimated according to equation (4) and (5), respectively.

u_(MEI+D,i)= mean dist(x〖,u〗_(MEI,Q(90) ) )=1/n*∑_(j=1)^n▒|x_j-x_(Q(u_MEI,0.9),i) | 				(4)

u_(MLI+D,i)= mean dist(x〖,u〗_(MLI,Q(90) ) )=1/n*∑_(j=1)^n▒|x_j-x_(Q(u_MLI,0.9),i) | 				(5)

where mean dist is the mean Euclidian distance, x_j are j-th coordinates of the known training data with n samples and x_(Q(u_MEI,0.9),i) and x_(Q(u_MLI,0.9),i) are the DS coordinates of the i-th candidate with a greater than 90 % quantile of the MEI or MLI utility. The MEI+D and MLI+D strategies aim at boosting the initial rounds of a SL run and hence were restricted to the 15 first iterations in the presented work. The utility was then calculated according to the MEI and MLI.





