# Auto Insurance Fraud Detection

## Business case

Claim related fraud is a huge problem in the insurance industry. It is quite complex and difficult to identify those unwanted claims. With Machine Learning Algorithm, I am trying to troubleshoot and help the General Insurance industry with this problem.

The data that I have is from Automobile Insurance. I will be creating a predictive model that predicts if an insurance claim is fraudulent or not. The answere between YES/NO, is a Binary Classification task.

A comparison study has been performed to understand which ML algorithm suits best to the dataset.

The raw dataset is in the file **"insurance_claim.csv"** which contains 1000 rows and 39 columns.

## Steps Involved

**Part 1** - Data Processing : Cleaning and Transforming Raw Data into the Understandable Format

**Part 2** - Profiling : Data profiling is the process of examining the data available from an existing information source (e.g. a database or a file) and collecting statistics or informative summaries about that data.

**Part 3** - Exploratory Data Analysis - EDA: Finding insights from the Data

**Part 4** - **Logistic regression, Decision Trees, and KNN**: Building the models and hyperparameter tuning, to find the best parameters.

**Part 5** - **Ensemble Learning and Random Forest**: Random forest and hyperparameter tuning, to find the best parameters.

**Part 6** - **Gradient Boosting and XGBoost**: XGBoost and hyperparameter tuning, to find the best parameters.

**Part 7** - **LightGBM and CatBoost**: Hyperparameter tuning, to find the best parameters.

**Part 7** - War between Boosting algorithm, Selecting the Best Model: XGBoost gives the best results on test dataset.

**Part 8** - Deploying the model on AWS Cloud

## Over-Sampling with SMOTE

One approach to addressing imbalanced datasets is to oversample the minority class. 
The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. 
Instead, new examples can be synthesized from the existing examples. 
This is a type of data augmentation for the minority class and is referred to as the **Synthetic Minority Oversampling Technique, or SMOTE** for short.

## Model Summary

![model_summary](https://user-images.githubusercontent.com/69073063/138944750-e64b36dd-d05c-4fb7-9e6f-fbac7e466a53.png)

## Commands to run the project locally

```scala
activate conda py38
python3 -m pipenv --python 3.8 shell 
docker build -t fraud_detection .
docker run -it --rm -p 9696:9696 fraud_detection
python3 predict-test-docker.py
``` 

## Commands to run the project from cloud service

```scala
activate conda py38
python3 -m pipenv --python 3.8 shell 
docker build -t fraud_detection .
docker run -it --rm -p 9696:9696 fraud_detection
python3 predict-test-cloud.py
``` 
