# Churn Reduction Framework: Customer Churn Prediction Project

This project is focused on analyzing customer data from a banking environment to predict customer churn using various machine learning techniques. By understanding the factors influencing customer churn, targeted strategies can be developed to improve customer retention. This analysis provides actionable insights for tailoring banking services to reduce churn rates.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Building and Evaluation](#model-building-and-evaluation)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributors](#contributors)

## Project Overview

Customer churn, the rate at which customers stop doing business with a company, is a critical metric impacting company revenue. In the banking sector, understanding the factors leading to customer churn is essential for developing effective customer retention strategies. This project aims to predict customer churn using machine learning models and identify key factors influencing customers' decisions to leave a bank.

## Dataset

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling/data). It contains 10,000 rows and 14 columns, providing comprehensive customer attributes such as credit score, geography, gender, age, tenure, balance, number of products, cardholder status, activity level, estimated salary, and churn status.

### Features:

- **RowNumber**: Unique number for each row in the dataset.
- **CustomerId**: Unique customer ID for all customers.
- **Surname**: Surname of the customers.
- **CreditScore**: Credit score of the customers.
- **Geography**: Location of the customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been associated with the bank.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of bank products the customer is using.
- **HasCrCard**: Whether the customer has a credit card.
- **IsActiveMember**: Whether the customer is an active member.
- **EstimatedSalary**: Estimated salary of the customer.
- **Exited**: Indicates whether the customer churned (target variable).

## Libraries Used

- **Pandas** and **NumPy**: For data manipulation and analysis.
- **Matplotlib**, **Seaborn**, and **Plotly**: For data visualization.
- **Scikit-learn**: For machine learning algorithms and model evaluation.
- **Imbalanced-learn**: For handling imbalanced datasets using techniques like SMOTE (Synthetic Minority Over-sampling Technique).

## Data Preprocessing

The integrity of data is crucial for predictive modeling. The data preprocessing steps included:

1. **Handling Missing Values**: Checked for null entries and confirmed their absence.
2. **Removing Duplicates**: Ensured the dataset was free from duplicate records.
3. **Dropping Irrelevant Columns**: Removed `RowNumber`, `CustomerId`, and `Surname` as they do not contribute to predicting churn.
4. **Renaming Target Variable**: Renamed `Exited` to `ChurnedOrNot` for clarity.
5. **Data Encoding**: Converted categorical variables into numerical format using techniques like One-Hot Encoding.

## Exploratory Data Analysis (EDA)

Various plots were generated to explore relationships between features and churn:

- **Histograms and Box Plots**: Visualized numerical variables such as Credit Score, Age, Balance, and Estimated Salary.
- **Bar Charts and Pie Charts**: Used to visualize categorical variables like Gender, Geography, Number of Products, Has Credit Card, and IsActiveMember.
- **Feature Distribution Based on Churn Status**: Analyzed distributions of features like Gender, Geography, and Active Membership based on churn status.

## Feature Engineering

New features were created to enhance model accuracy:

- **Categorizing Numerical Variables**: Transformed continuous variables like `NumOfProducts` and `Balance` into categorical variables to simplify the data and improve model performance.

## Model Building and Evaluation

Multiple machine learning models were built and evaluated using various performance metrics:

### Logistic Regression

- **Metrics**: Accuracy: 72.05%, F1 Score: 0.7205, AUC: 0.76.
- **Feature Importance**: Identified key features like `Tot_Products_Two Products` and `IsActiveMember` as significant predictors.
- **Confusion Matrix and ROC Curve**: Provided insights into the model's predictive performance.

### Decision Tree

- **Metrics**: Training Accuracy: 89.15%, Testing Accuracy: 83.4%, F1 Score: 0.834.
- **Feature Importance**: Visualized key features influencing the model's predictions.
- **Confusion Matrix and ROC Curve**: Evaluated model performance in distinguishing between churn and non-churn events.

### Random Forest

- **Metrics**: Training Accuracy: 90.61%, Testing Accuracy: 84.2%, F1 Score: 0.84, AUC: 0.86.
- **Feature Importance**: Highlighted the most influential features for predicting churn.
- **Confusion Matrix and ROC Curve**: Demonstrated strong discriminative ability.

## Results

The Random Forest model emerged as the most effective for customer churn prediction, with the highest testing accuracy of 84.2% and an AUC of 0.86. It successfully balances the trade-off between true positive rate and false positive rate, making it a robust choice for predicting customer churn.

## Conclusion

The project demonstrates the importance of data-driven approaches in enhancing customer retention efforts in the banking sector. The Random Forest model proved to be the best performer, offering reliable predictions for customer churn. Further exploration into advanced modeling techniques and deeper feature engineering could provide even greater insights and improved outcomes.

## Contributors

- **Aravind Swamy** - [swamy.ar@northeastern.edu](mailto:swamy.ar@northeastern.edu)
- **Jayasurya Jagadeesan** - [jegadeesan.j@northeastern.edu](mailto:jegadeesan.j@northeastern.edu)
- **Keshika Arunkumar** - [arunkumar.k@northeastern.edu](mailto:arunkumar.k@northeastern.edu)
- **Samyuktha Kapoor** - [rajeshkapoor.s@northeastern.edu](mailto:rajeshkapoor.s@northeastern.edu)
- **Sruthi Gandla** - [gandla.s@northeastern.edu](mailto:gandla.s@northeastern.edu)
