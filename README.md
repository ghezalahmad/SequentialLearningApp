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

where mean dist is the mean Euclidian distance, x_j are j-th coordinates of the known training data with n samples and x_(Q(u_MEI,p),i) and x_(Q(u_MLI,p),i) are the DS coordinates of the i-th candidate with a greater than p % quantile of the MEI or MLI utility. The MEI+D and MLI+D strategies aim at boosting the initial rounds of a SL run and hence were restricted to the 15 first iterations in the presented work. The utility was then calculated according to the MEI and MLI.

### Benchmarking SL against a Random Process (RP)
Although, SL is based on ML methods, classical error-based ML benchmarks typically do not apply in this context. This is because the target of SL in materials discovery is to find a candidate with - depending on the property - maximum or minimum value of a said property. In a reasonably set scenario, this goal is always achieved with zero error and is merely a matter of iterations. Although this comparison is somewhat odd from a mathematical point of view, it underscores the fact that the focus here is on the effort required to reach this threshold as a measure of performance. A common metric is the required number of experiments until a set target is reached. 
To determine the performance of SL methods, it is common to use simulated experiments where the ground truth labels for all data points are already known. Initially, only a small fraction is provided to the SL algorithm (although more training data would be available). This is extended with one new data point from the remainder of the available data at each iteration. It is investigated which approach requires the least amount of data to achieve the goal. Approaches that require less data simply lead to faster success in laboratory practice. 
Thus, the goal is not to actually discover new materials using all available data, but to validate material discovery methods for scenarios where fewer labels are known (e.g., for new materials). In this approach, the generalizability is statistically demonstrated by quantifying the performance of SL methods under randomized initial conditions and then expressing it, for example, as a mean value and standard deviation. This allows meaningful comparisons between different SL approaches. To generate randomized initial conditions in an in-progress experimental study would require significant additional effort and is unrealistic in most cases. Therefore, comparisons of performance and repeatability between different SL methods in actual material discovery are usually not possible. 
This approach also differs from the classical ML approach, where generalizability is demonstrated on retained test data. However, this luxury is often not afforded in experimental science, where data are extremely limited due to costly acquisition.
SL is commonly compared against a Random Process (RP) (i.e., acting without a strategy and model) as a baseline benchmark. RPs consider each candidate equally likely to succeed (uniform distribution). The average number of draws necessary to find the maximum target property is 50 % of the given candidates, respectively. This is the benchmark against which SL competes. 
Despite the fact that this benchmark is often surpassed by SL, a significant use of SL cannot be found in practice. One reason for this may be the significantly higher effort that is caused by the sequentialization of the experimental procedure in SL. This means that from a purely functional point of view, RP can produce the desired results faster if the parallelization of experiments is more effective. In view of this situation, it is worthwhile to include further parameters for the consideration of the usefulness of SL in practice. 
The specific value of the target threshold T (i.e., the property value to be exceeded) inherently affects the iteration required; the smaller T, the fewer iterations are required for SL to succeed. From a practical perspective, relatively small deviations of the highest cement strengths contradict a special significance of a unique strength value as the target (especially considering the aleatory uncertainties of this value). To accelerate experimental progress, one can argue to reduce T, to a value that lies in the upper quantile of strengths (e.g., T≥f_((c,90%))) without losing much significance of the results.
Furthermore, the aspired success rate determines the number of experiments required. The relationship is simply: the higher a desired success rate, the more experiments are needed. The performance of SL at a certain success rate can be empirically determined as the quantile of the required draws from multiple SL runs. In the laboratory practice, the required success rate is expected to be much higher than the 50 % rate, which is, as mentioned above, the typical benchmark for SL. 
The relationship between success rate and target threshold can be described analytically for RP as the hypergeometric cumulative distribution according to the following equation:

p(one success |n_max  draws)=∑_(n=1)^(n_max)▒(K/1)((N-K)/(n-(k=1)))/((N/n) )						     (6)

where p corresponds to the success rate, N is the size of the population, K is the number of items with the desired characteristic in the population, and n is the number of samples drawn. The threshold of success T can be defined in terms of the parameters M and x. According to equation (6), the success rate p has a non-linear relationship with the required draws for the case of multiple targets (where M>1), i.e., the before mentioned rule that a 50 % success rate requires 50 % of the possible experiments holds not for those cases. Instead, much less data is required. The exact amount further depends on the size of the population x where a greater x leads to fewer required draws n. 
This means that RP becomes a much tougher benchmark when T can be reduced to include more successful candidates.






