# Loan Approval Data Analysis: Building and Evaluating Logistic Regression and KNN Models for Predicting Loan Approval

This repository contains the analysis of a loan approval dataset aimed at building predictive models to assess whether a loan applicant is likely to be approved or denied. The dataset includes 20,000 records containing personal and financial data, such as demographic information, credit history, employment status, income levels, existing debt, and other relevant metrics. The analysis primarily focuses on training classification models and evaluating their performance in predicting loan approval status.

## Data Cleaning and Preprocessing

### Task 1: Clean the Datasets
- **Handling Missing Values**: 
    - Columns with over 50% missing values are removed.
    - Rows with missing categorical data are dropped.
    - Missing values in numerical columns are imputed with the column's average value.
  
- **Handling Categorical Data**: 
    - Columns with unique categorical values are removed.
    - One-hot encoding is applied to convert categorical variables into numerical ones.

## Task 2: Logistic Regression Model

### Step 2.1: Data Splitting
- The dataset is split into training and testing sets with an 80/20 ratio, using a fixed random state (48724432) for reproducibility.

### Step 2.2: Logistic Regression Model
- A logistic regression model is built and trained on the dataset.
- Performance metrics such as accuracy and F1-score are reported for both training and testing data.
- Model overfitting is analyzed by comparing training and testing results, with justification.

### Step 2.3: Recursive Feature Elimination (RFE)
- The RFE technique is applied to identify the most effective features for the model.
- A line chart is used to visualize performance changes with respect to the number of eliminated features.

## Task 3: KNN Classification Model

### Step 3.1: 1-NN Classifier
- A 1-NN classifier is built using the features selected from RFE.
- Accuracy and F1-score are calculated for both training and testing data, with performance compared to detect overfitting.

### Step 3.2: Grid Search and Cross-Validation
- Grid search with 5-fold cross-validation is used to select the optimal number of neighbors (K) for the KNN classifier.
- Performance metrics (accuracy and F1-score) are reported for the best K.

### Step 3.3: Distance Metrics Comparison
- The KNN model is tested with three different distance metrics: Euclidean (L2), Manhattan (L1), and Cosine.
- A bar chart is used to compare the performance metrics (accuracy and F1-score) for each distance metric, determining the best and worst-performing metrics.

## Key Findings:

1. **Logistic Regression**:
    - The logistic regression model performs reasonably well, with an accuracy of 90% on both training and testing data.
    - The model is not overfitting, as the performance difference between training and testing sets is minimal.

2. **KNN Classifier**:
    - The 1-NN classifier showed perfect performance on the training set (accuracy = 1.00), but struggled with generalization to new data (accuracy = 0.79), indicating overfitting.
    - The optimal number of neighbors (K) found through grid search is 29, with an accuracy of 84% on the test set.

3. **Distance Metric Comparison**:
    - The Manhattan distance metric provided the best results with an accuracy of 84%, while the Cosine distance metric performed the worst with the lowest accuracy.

## Conclusion:
The analysis demonstrates that logistic regression is effective in predicting loan approval, with minimal overfitting. However, KNN, particularly with the 1-NN classifier, shows overfitting, highlighting the need for better model tuning. The Manhattan distance metric is found to perform the best among the three tested distance metrics.

