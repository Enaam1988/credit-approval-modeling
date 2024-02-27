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

1. **Exploratory Data Analysis (EDA):** Analyze the distribution of features, explore correlations, and identify patterns.
2. **Data Preparation:** Preprocess the dataset, encode categorical variables.
3. **Modeling:** Train machine learning models to predict credit card approval outcomes.
4. **Evaluation:** Assess model performance and identify the most influential factors.
5. **Insights and Recommendations:** Extract actionable insights and provide recommendations for credit card issuers.

## Project Overview

### Exploratory Data Analysis (EDA) and Imbalance Observation

Before modeling, we conducted EDA to understand the dataset. Notably, we observed an imbalance between approved and unapproved applications, prompting further investigation prior to data cleaning.

### Feature Descriptions ###

| Column Name          | Description                                                      |
|----------------------|------------------------------------------------------------------|
| Applicant_ID         | Unique identifier for each applicant                              |
| Applicant_Gender     | Gender of the applicant (M = Male, F = Female)                    |
| Owned_Car            | Indicates if the applicant owns a car (1 = Yes/Have; 0 = No/Don't Have) |
| Owned_Realty         | Indicates if the applicant owns real estate (1 = Yes/Have; 0 = No/Don't Have) |
| Total_Children       | Total number of children the applicant has                        |
| Total_Income         | Total income of the applicant                                     |
| Income_Type          | Type of income (Working, Commercial associate, Other)             |
| Education_Type       | Level of education of the applicant (e.g., High School, Bachelor) |
| Family_Status        | Marital status or family status of the applicant                  |
| Housing_Type         | Type of housing the applicant resides in (e.g., House/Apartment)  |
| Owned_Mobile_Phone   | Indicates if the applicant owns a mobile phone (1 = Yes/0 = No)    |
| Owned_Work_Phone     | Indicates if the applicant owns a work phone (1 = Yes/0 = No)      |
| Owned_Phone          | Indicates if the applicant owns a phone (1 = Yes/0 = No)           |
| Owned_Email          | Indicates if the applicant owns an email address (1 = Yes/0 = No)  |
| Job_Title            | Title or position of the applicant's job                          |
| Total_Family_Members | Total number of family members in the applicant's household       |
| Applicant_Age        | Age of the applicant                                              |
| Years_of_Working     | Number of years the applicant has been working                    |
| Total_Bad_Debt       | Total amount of bad debt accumulated by the applicant             |
| Total_Good_Debt      | Total amount of good debt accumulated by the applicant            |
| Status               | Approval status of the credit card application (0: Not Approved, 1: Approved) |

 
### Data Engineering Features

We incorporated `sklearn.preprocessing` to import OneHotEncoder for categorical features and StandardScaler for numerical features like number of children, income, and age. This preprocessing facilitated data transformation, providing valuable insights into applicant demographics and financial profiles, which guided our modeling approach.

### Feature Selection

Out of the initial 21 features, we selected 14, including the target variable "credit card approval," after dropping 5 features that had minimal impact on model performance. Notably, we omitted the "total children" feature in favor of "total family number" due to their high correlation.

### Modeling and Classification Performance
#### logistic regression model

| Metric                    | Value    |
|---------------------------|----------|
|**Balanced Accuracy Score**| 0.7      |

#### Confusion Matrix

 |      | Predicted | |
|:----:|:---------:|---:|
|      |     0     |  1 |
|   0  |     10    |  15|
|   1  |     0     |5001|
  

##### Classification Report 

|                         | Precision | Recall | F1-Score | Support |
|-------------------------|-----------|--------|----------|---------|
| **Not Approved (0)**    |    1.00   |  0.40  |   0.57   |    25   |
| **Approved (1)**        |    1.00   |  1.00  |   1.00   |  5001   |
| **Accuracy**            |           |        |   1.00   |   5026  |
| **Macro Avg**           |   1.00    |  0.70  |   0.78   |   5026  |
| **Weighted Avg**        |   1.00    |  1.00  |   1.00   |   5026  |


The logistic regression model's performance reveals that it achieves a high level of accuracy and precision for approved credit card applications. However, it demonstrates a significant limitation in correctly identifying unapproved applications. Specifically, the model's recall rate for unapproved applications is relatively low, indicating that it often fails to detect these instances. This deficiency is highlighted by the balanced accuracy score, which, while reasonably high overall, suggests room for improvement, particularly in capturing true negatives. Overall, while the model is effective in predicting approved applications, it requires enhancement to better identify and classify unapproved ones.
### Lazy Predict

We used Lazy Predict, which automates training and evaluating multiple machine learning models on a dataset, providing quick performance metrics for comparison and selection of the most suitable algorithms. We applied Bagging Classifier and XGB Classifier, yielding the following results:
#### Bagging Classifier
| Metric                    | Value    |
|---------------------------|----------|
|**Balanced Accuracy Score**| 0.98     |

#### Confusion Matrix  

|      | Predicted | |
|:----:|:---------:|---:|
|      |     0     |  1 |
|   0  |     23    |  2 |
|   1  |     0     |5001|
  
##### Classification Report 

|                     | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| **Not Approved (0)**|    1.00   |  0.96  |   0.98   |    25   |
| **Approved (1)**    |    1.00   |  1.00  |   1.00   |  5001   |
| **Accuracy**        |           |        |   1.00   |  5026   |
| **Macro Avg**       |    1.00   |  0.98  |   0.99   |  5026   |
| **Weighted Avg**    |    1.00   |  1.00  |   1.00   |  5026   |

  
The Bagging Classifier demonstrates outstanding performance with a balanced accuracy score of 0.98. It effectively identifies both approved and unapproved credit card applications, as indicated by the high precision, recall, and F1-score values. The confusion matrix shows only one misclassified instance of a not approved application out of 25, while correctly classifying all 5001 approved applications. This exceptional performance underscores the efficacy of the Bagging Classifier in predicting credit card approval outcomes.

**XGB Classifier**

| Metric                   | Value    |
|--------------------------|----------|
| Balanced Accuracy Score  | 0.96     |

#### Confusion Matrix

|      | Predicted | |
|:----:|:---------:|---:|
|      |     0     |  1 |
|   0  |     24    |  1 |
|   1  |     0     |5001|
  
##### Classification Report 

|                     | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| **Not Approved (0)**|    1.00   |  0.92  |   0.96   |    25   |
| **Approved (1)**    |    1.00   |  1.00  |   1.00   |  5001   |
| **Accuracy**        |           |        |   1.00   |  5026   |
| **Macro Avg**       |    1.00   |  0.96  |   0.98   |  5026   |
| **Weighted Avg**    |    1.00   |  1.00  |   1.00   |  5026   |

  
The XGB Classifier achieves a balanced accuracy score of 0.96, indicating high performance in predicting credit card approval outcomes. With a precision, recall, and F1-score of 1.00 for approved applications, the model accurately identifies the vast majority of successful credit card applications. However, for unapproved applications, the recall drops to 0.92, suggesting a slight decrease in performance compared to the Bagging Classifier. It's important to note that the support for not approved applications is only 25 instances, which may raise concerns about the model's generalization to this minority class. Nonetheless, the model demonstrates excellent overall performance, providing valuable insights for credit card issuers.
## Dataset Balancing

Achieving balance is crucial for optimizing model performance. We balanced the dataset by oversampling the minority class, undersampling the majority class, and employing techniques like RandomOverSampler, ensuring improved model performance and generalization.

To address the imbalance in a dataset, one approach is to oversample the minority class by generating additional instances using the RandomOverSampler technique.

### Logistic Regression Model Performance (Resampled)

| Metric                   | Value    |
|---------------------------|----------|
|**Balanced Accuracy Score**| 0.9972   |


#### Confusion Matrix

|      | Predicted | |
|:----:|:---------:|---:|
|      |     0     |  1  |
|   0  |    20006  |  1  |
|   1  |     111   |19895|

#### Classification Report

|                     | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
|**Not Approved (0)** |    0.99   |  1.00  |   1.00   |  20006  |
|**Approved (1)**     |    1.00   |  0.99  |   1.00   |  20006  |
|**Accuracy**         |           |        |   1.00   |  40012  |
| **Macro Avg**       |    1.00   |  1.00  |   1.00   |  40012  |
| **Weighted Avg**    |    1.00   |  1.00  |   1.00   |  40012  |

- The balanced accuracy score is exceptionally high, indicating that the model performs well in predicting both classes, considering the dataset is balanced after resampling.
- The confusion matrix shows that out of 20,117 instances of the negative class (0), the model correctly predicted 20,006 instances, and out of 19,906 instances of the positive class (1), the model correctly predicted 19,895 instances. There are very few misclassifications.
- The classification report further confirms the excellent performance of the model, with high precision, recall, and F1-score values for both classes. The accuracy, macro avg, and weighted avg scores are all 1.00, indicating perfect classification performance.

Overall, the model exhibits outstanding performance on the resampled dataset, with high accuracy and precision in predicting credit card approval outcomes for both approved and not approved applications.

**Exceptional Performance of XGB Classifier on Resampled Training Data**

The XGB Classifier shows outstanding performance on the resampled training data, attaining a perfect balanced accuracy score of 1.0. The confusion matrix underscores its effectiveness, correctly classifying all instances of both classes (0 and 1). Additionally, the classification report confirms exceptional precision, recall, and F1-score values for both classes, indicating the model's ability to accurately identify approved and not approved credit card applications in the resampled training dataset. Overall, the XGB Classifier demonstrates impeccable performance, showcasing its capacity to generalize well and make precise predictions.

**Exceptional Performance of Bagging Classifier on Resampled Training Data**

The Bagging Classifier demonstrates impeccable performance on the resampled training data, achieving a perfect balanced accuracy score of 1.0. This signifies that the model accurately predicts both approved and not approved credit card applications without any misclassifications. The confusion matrix confirms that all instances in both classes (0 and 1) are correctly classified, indicating the model's robustness in capturing the underlying patterns in the data. Moreover, the classification report further emphasizes the model's exceptional precision, recall, and F1-score values for both classes, showcasing its ability to generalize well and make accurate predictions. Overall, the Bagging Classifier exhibits flawless performance on the resampled training data, highlighting its effectiveness in credit card approval prediction.

## Conclusions

**Model Performance:** The machine learning models, including Logistic Regression, Bagging Classifier, and XGB Classifier, demonstrate high accuracy and precision in predicting credit card approval outcomes. They achieve balanced accuracy scores close to 1.0, indicating robust performance.

**Data Imbalance:** The dataset initially exhibits class imbalance, with a significantly higher number of approved applications compared to unapproved ones. Resampling techniques such as oversampling and undersampling effectively address this issue, resulting in improved model performance and generalization.

**Feature Importance:** Certain features, such as income level, education type, and family status, appear to significantly influence credit card approval decisions. Analyzing feature importance can provide insights into the factors driving approval outcomes and help optimize decision-making processes.

## Recommendations

**Model Deployment:** Deploy the trained machine learning models to automate credit card approval processes. By integrating these models into existing systems, credit card issuers can streamline decision-making and enhance efficiency.

**Continuous Monitoring:** Regularly monitor model performance and retrain the models as necessary to adapt to changing trends and patterns in credit card applications. This ensures that the models remain accurate and reliable over time.

**Interpretability:** Enhance the interpretability of the models by providing explanations for their predictions. This transparency can foster trust among stakeholders and regulatory bodies and facilitate compliance with industry regulations.

**Customer Segmentation:** Utilize clustering techniques to segment customers based on their credit profiles and preferences. This segmentation can enable targeted marketing strategies and personalized product offerings, ultimately improving customer satisfaction and retention.

## Contact

If you have any questions or need further information, please contact us:

- Enaam Hamad: [emoemo1988@yahoo.com](mailto:emoemo1988@yahoo.com)
- Bisma Jalili: [bismajalili@gmail.com](mailto:bismajalili@gmail.com)
- Shaza Abdalsalam: [shazaaali94@gmail.com](mailto:shazaaali94@gmail.com)
- Zainab Arif: [zainab.arif1998@gmail.com](mailto:zainab.arif1998@gmail.com)

