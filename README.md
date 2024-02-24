# Credit Card Approval Analysis

Credit cards play a pivotal role in modern finance, offering convenience and flexibility to consumers globally. This project analyzes a dataset from Kaggle to gain insights into the credit card approval process, with the aim of optimizing approval processes and managing risk effectively.

## Overview

This project explores patterns and trends in credit card approval data, examining demographic and financial indicators to uncover key factors influencing approval rates. By extracting actionable insights from the data, the project aims to improve decision-making processes for credit card issuers.

## Dataset

The dataset used in this project is sourced from Kaggle and provides information on credit card applicants, including demographic details, financial indicators, and approval outcomes. You can find the dataset [here](https://www.kaggle.com/datasets/caesarmario/application-data).

## Objectives

- Analyze approval rates and trends
- Identify key factors influencing credit card approval
- Optimize approval processes
- Manage risk effectively
- Analyze approval status and its impact on credit card application outcomes

## Tools and Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn for data manipulation and visualization
- Scikit-learn for machine learning modeling
- Jupyter Notebook for development and analysis
- Apache Spark for distributed data processing (used in Google Colab Notebook)

### Google Colab Notebook

The project was developed using Google Colab Notebook, a cloud-based Python environment that provides free access to GPUs and TPUs for accelerated computing. Google Colab offers seamless integration with Google Drive and allows collaborative editing and sharing of notebooks.

### Apache Spark

Apache Spark was utilized for distributed data processing tasks. The SparkSession object was created to interact with Spark APIs and perform data analysis and transformations on large-scale datasets.

## Methodology

1. **Data Preparation:** Preprocess the dataset, encode categorical variables.
2. **Exploratory Data Analysis (EDA):** Analyze the distribution of features, explore correlations, and identify patterns.
3. **Modeling:** Train machine learning models to predict credit card approval outcomes.
4. **Evaluation:** Assess model performance and identify the most influential factors.
5. **Insights and Recommendations:** Extract actionable insights and provide recommendations for credit card issuers.

## Project Overview

### Exploratory Data Analysis (EDA)

Before modeling, we conducted EDA to understand the dataset. Notably, we observed an imbalance between approved and unapproved applications, prompting further investigation prior to data cleaning.

### Data Engineering Features

We incorporated `sklearn.preprocessing` to import OneHotEncoder for categorical features and StandardScaler for numerical features like number of children, income, and age. This preprocessing facilitated data transformation, providing valuable insights into applicant demographics and financial profiles, which guided our modeling approach.

### Feature Selection

Out of the initial 21 features, we selected 14, including the target variable "credit card approval," after dropping 5 features that had minimal impact on model performance. Notably, we omitted the "total children" feature in favor of "total family number" due to their high correlation.

### Modeling and Classification Performance
#### logistic regression model

| Metric                   | Value    |
|--------------------------|----------|
| Balanced Accuracy Score  | 0.7      |

#### Confusion Matrix  

| Actual|          | Predicted |
| --- | --- |
|                  |  0        |  1        |
| 0                |  10       |  15       |
| 1                |  0        |  5001     |

##### Classification Report 

|                         | Precision | Recall | F1-Score | Support |
|-------------------------|-----------|--------|----------|---------|
| Not Approved (0)        |    1.00   |  0.40  |   0.57   |    25   |
| Approved (1)            |    1.00   |  1.00  |   1.00   |  5001   |
| Accuracy                |           |        |   1.00   |   5026  |
| Macro Avg               |   1.00    |  0.70  |   0.78   |   5026  |
| Weighted Avg            |   1.00    |  1.00  |   1.00   |   5026  |


The logistic regression model's performance reveals that it achieves a high level of accuracy and precision for approved credit card applications. However, it demonstrates a significant limitation in correctly identifying unapproved applications. Specifically, the model's recall rate for unapproved applications is relatively low, indicating that it often fails to detect these instances. This deficiency is highlighted by the balanced accuracy score, which, while reasonably high overall, suggests room for improvement, particularly in capturing true negatives. Overall, while the model is effective in predicting approved applications, it requires enhancement to better identify and classify unapproved ones.
### Lazy Predict

We used Lazy Predict, which automates training and evaluating multiple machine learning models on a dataset, providing quick performance metrics for comparison and selection of the most suitable algorithms. We applied Bagging Classifier and XGB Classifier, yielding the following results:
#### Bagging Classifier
| Metric                   | Value    |
|--------------------------|----------|
| Balanced Accuracy Score  | 0.98     |

#### Confusion Matrix  
|       | Predicted |
|-------|-----------|
|       |    0      |    1      |
|-------|-----------|
|   0   |    24     |    1      |
|   1   |    0      |   5001    |


|                     | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| **0 (Not Approved)**|    1.00   |  0.96  |   0.98   |    25   |
| **1 (Approved)**    |    1.00   |  1.00  |   1.00   |  5001   |
| **Accuracy**        |           |        |   1.00   |  5026   |
| **Macro Avg**       |    1.00   |  0.98  |   0.99   |  5026   |
| **Weighted Avg**    |    1.00   |  1.00  |   1.00   |  5026   |

  
The Bagging Classifier demonstrates outstanding performance with a balanced accuracy score of 0.98. It effectively identifies both approved and unapproved credit card applications, as indicated by the high precision, recall, and F1-score values. The confusion matrix shows only one misclassified instance of a not approved application out of 25, while correctly classifying all 5001 approved applications. This exceptional performance underscores the efficacy of the Bagging Classifier in predicting credit card approval outcomes.

**XGB Classifier**

|                     | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| **0 (Not Approved)**|    1.00   |  0.92  |   0.96   |    25   |
| **1 (Approved)**    |    1.00   |  1.00  |   1.00   |  5001   |
| **Accuracy**        |           |        |   1.00   |  5026   |
| **Macro Avg**       |    1.00   |  0.96  |   0.98   |  5026   |
| **Weighted Avg**    |    1.00   |  1.00  |   1.00   |  5026   |

  
The XGB Classifier achieves a balanced accuracy score of 0.96, indicating high performance in predicting credit card approval outcomes. With a precision, recall, and F1-score of 1.00 for approved applications, the model accurately identifies the vast majority of successful credit card applications. However, for unapproved applications, the recall drops to 0.92, suggesting a slight decrease in performance compared to the Bagging Classifier. It's important to note that the support for not approved applications is only 25 instances, which may raise concerns about the model's generalization to this minority class. Nonetheless, the model demonstrates excellent overall performance, providing valuable insights for credit card issuers.
## Dataset Balancing

Achieving balance is crucial for optimizing model performance. We balanced the dataset by oversampling the minority class, undersampling the majority class, and employing techniques like RandomOverSampler, ensuring improved model performance and generalization.

To address the imbalance in a dataset, one approach is to oversample the minority class by generating additional instances using the RandomOverSampler technique.

### Logistic Regression Model Performance (Resampled)
| Metric                   | Value    |
|--------------------------|----------|
| Balanced Accuracy Score  | 0.9972   |
|--------------------------|----------|
| Confusion Matrix         |          |
|                          | Predicted|
|                          |  0  |  1  |
|--------------------------|------|------|
| Actual    |  0  | 20006|   0  |
|           |  1  |  111 | 19895|
|--------------------------|------|------|
| Classification Report    |          |
|                          | Precision | Recall | F1-Score | Support |
|--------------------------|-----------|--------|----------|---------|
| 0                        |    0.99   |  1.00  |   1.00   |  20006  |
| 1                        |    1.00   |  0.99  |   1.00   |  20006  |
|--------------------------|-----------|--------|----------|---------|
| Accuracy                 |                     |          |          |         |
|--------------------------|---------------------|----------|----------|---------|
|                          |                     |          |          |  40012  |
