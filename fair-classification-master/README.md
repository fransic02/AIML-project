# Fairness in Classification in Compass scores dataset

 
This repository provides a logistic regression implementation in python to apply the fair classification mechanisms introduced by the Max Planck Institute for Software Systems in the <a href="http://arxiv.org/abs/1507.05259" target="_blank">AISTATS'17</a> papers.

The <a href="http://arxiv.org/abs/1507.05259" target="_blank">AISTATS'17 paper</a> [1]  proposes mechanisms to make classification outcomes free of disparate impact, that is, to ensure that similar fractions of people from different demographic groups (e.g., males, females) are accepted (or classified as positive) by the classifier. 

#### Using the code

The fairness.py file in the fair-classification-master directory executes a classification task on the Compas scores dataset by using the sklearn DecisionTreeClassifier.

In the compas_disparate_impact directory there are the implementation of the proposed fairness mechanism.
Specifically, there are two different main files:
- main.py: applies the fairness mechansim on the Compas scores dataset by using the real data in the trainig set and test set
- main_synthetic.py: applies the fairness mechansim on the Compas scores dataset by using synthetic data in the trainig set and real data in the test set

#### References
<a href="http://arxiv.org/abs/1507.05259" target="_blank">Fairness Constraints: Mechanisms for Fair Classification</a> <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi. <br>
20th International Conference on Artificial Intelligence and Statistics (AISTATS), Fort Lauderdale, FL, April 2017.