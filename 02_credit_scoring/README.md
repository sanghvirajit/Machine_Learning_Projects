# Credit Score Analysis

A credit scoring model is a tool that is typically used in the decision-making process of accepting or rejecting a loan. A credit scoring model is the result of a statistical model which, based on information about the borrower (e.g. age, number of previous loans, etc.), allows one to distinguish between "good" and "bad" loans and give an estimate of the probability of default.

Loan Default: In finance, default is failure to meet the legal obligations of a loan, for example when a home buyer fails to make a mortgage payment, or when a corporation or government fails to pay a bond which has reached maturity.

The raw dataset is in the file **"CreditScoring.csv"** which contains 4455 rows and 14 columns:

<table>
<tbody>
<tr><td><b>1  Status</b></td> <td>credit status</td></tr>
<tr><td><b>2  Seniority</b></td> <td>job seniority (years)</td></tr>
<tr><td><b>3  Home</b></td> <td>type of home ownership</td></tr>
<tr><td><b>4  Time</b></td> <td>time of requested loan</td></tr>
<tr><td><b>5  Age</b></td> <td>client's age </td></tr>
<tr><td><b>6  Marital</b></td> <td>marital status </td></tr>
<tr><td><b>7  Records</b></td> <td>existance of records</td></tr>
<tr><td><b>8  Job</b></td> <td>type of job</td></tr>
<tr><td><b>9  Expenses</b></td> <td> amount of expenses</td></tr>
<tr><td><b>10 Income</b></td> <td> amount of income</td></tr>
<tr><td><b>11 Assets</b></td> <td> amount of assets</td></tr>
<tr><td><b>12 Debt</b></td> <td> amount of debt</td></tr>
<tr><td><b>13 Amount</b></td> <td> amount requested of loan</td></tr>
<tr><td><b>14 Price</b></td> <td> price of good</td></tr>
</tbody>
</table>

# Steps Involved

* Part 1 - **Data Processing** : Cleaning and Transforming Raw Data into the Understandable Format
* Part 2 - **Profiling** : Data profiling is the process of examining the data available from an existing information source (e.g. a database or a file) and collecting statistics or informative summaries about that data.
* Part 3 - **Exploratory Data Analysis - EDA**: Finding insights from the Data
* Part 4 - **Decision Trees**: Building Decision tree model and hyperparameter tuning, to find the best parameters.
* Part 5 - **Ensemble Learning and Random Forest**: Random forest and hyperparameter tuning, to find the best parameters.
* Part 6 - **Gradient Boosting and XGBoost**: XGBoost and hyperparameter tuning, to find the best parameters.
* Part 7 - **Selecting the Best Model**: XGBoost gives the best results on validation dataset
* Part 8 - **Deploying the model on AWS Cloud**

# Model Results

1. Decision Tree
  
    train auc:  0.765
  
    val auc:  0.679

2. Random Forest
  
    train auc:  0.822
  
    val auc:  0.669
    
3. **XGBoost**
  
    train auc:  **0.942**
  
    val auc:  **0.835**
    
# Commands to run the project locally

```scala
activate conda py38
python3 -m pipenv --python 3.8 shell 
docker build -t credit_scoring .
docker run -it --rm -p 9696:9696 credit_scoring
python3 predict-test.py
``` 
