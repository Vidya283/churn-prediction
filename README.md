# Customer Churn Prediction
customer churn prediction project using ML

This project aims to predict whether a customer will churn (i.e., leave a service) using machine learning models. Understanding churn behavior helps businesses retain customers and reduce revenue loss.

---

## 📂 Project Overview

- **Problem**: Churn leads to significant revenue loss for subscription-based businesses. Early detection allows for retention strategies.
- **Objective**: Use customer demographic and activity data to predict churn with various ML algorithms and compare performance.

---

## 📊 Dataset

- **Source**: [Kaggle](https://www.kaggle.com/) — /kaggle/input/churn-modelling/Churn_Modelling.csv
- **Features**:
  - CreditScore, Age, Tenure, Balance, NumOfProducts, IsActiveMember, etc.
  - Geography (Germany/Spain/France - one-hot encoded)
  - Target Variable: `Exited` (1 = churned, 0 = stayed)

---

## 🧼 Data Preprocessing

- Checked for missing values (none found)
- Verified data types
- Remove unnecessary cols
- Encoded categorical variables using one-hot encoding
- Ensured consistent formatting



## 🔎 Exploratory Data Analysis (EDA)

Performed visual exploration of customer behavior and churn patterns.

### 🔹 Key EDA Visuals and Insights:

- **Churn Distribution**  
  Count plot showing the imbalance between churned and retained customers.  
  📌 *Most customers did not churn (class imbalance exists).*

- **Churn Rate by Age**  
  Histogram showing that churn is more likely among older customers.  
  📌 *Customers above 50 show a noticeably higher churn rate.*

- **Churn by Country (Geography)**  
  Bar plot comparing churn rates across countries.  
  📌 *Customers from Germany had the highest churn rate, while Spain had the lowest.*

- **Correlation Matrix**  
  Heatmap visualizing correlation between numerical features.  
  📌 *No extremely high correlations. CreditScore and Age show mild correlation with churn.*

<details>
<summary>📈 Click to expand visualizations</summary>

![Churn Distribution](images/churn_distribution.png)  
![Age vs Churn](images/churn_by_age.png)  
![Churn by Country](images/churn_by_country.png)  
![Correlation Heatmap](images/heatmap.png)

</details>

---

## 🏗️ Feature Engineering

To enhance predictive power, new features were engineered based on domain insights:

| Feature Name            | Description |
|-------------------------|-------------|
| `AgeGroup`              | Binned age into categorical groups: **18–30**, **31–40**, **41–50**, **51+** |
| `HighBalance`           | Binary flag: `1` if Balance > ₹100,000; otherwise `0` |
| `Balance_to_Products`   | Ratio of account balance to number of products held (`Balance / NumOfProducts`) |
| `BalanceSalaryRatio`    | Customer's balance relative to their salary (`Balance / EstimatedSalary`) |
| `CreditScoreAgeRatio`   | Credit score per year of age (`CreditScore / Age`) |

### 🧠 Why These Features Matter:

- **AgeGroup**: Age often impacts churn, but the effect may be non-linear.
- **HighBalance**: High-value customers might be more sensitive or harder to retain.
- **Balance-to-Product**: High balance with fewer products could indicate disengagement.
- **BalanceSalaryRatio**: Helps differentiate high earners with low account usage.
- **CreditScoreAgeRatio**: Normalizes credit score relative to age — a proxy for risk assessment.

These new features improved model interpretability and slightly boosted performance across most classifiers.

---

### 📌 Scaling Approach:

- Scaling was applied **only to the training data**, and the same transformation was used on the test set (to prevent data leakage).
- Used `StandardScaler` from `sklearn.preprocessing`, which scales features to have **zero mean** and **unit variance**.

### 🔄 Scaled Features Included:
- `CreditScore`
- `Age`
- `Tenure`
- `Balance`
- `EstimatedSalary`
- Engineered features like `BalanceSalaryRatio`, `CreditScoreAgeRatio`

> ⚠️ **Note**: Scaling was applied only for models sensitive to feature magnitude, such as:
> - Logistic Regression  
> - Support Vector Machine  
> - K-Nearest Neighbors  

Tree-based models like **Random Forest** and **Gradient Boosting** do not require feature scaling.

---

## 🤖 Model Building

Trained and evaluated the following models:

| Model                       | Accuracy | ROC AUC |
|----------------------------|----------|---------|
| Logistic Regression        | 0.82     | 0.76    |
| Support Vector Machine     | 0.82     | 0.76    |
| K-Nearest Neighbors        | 0.82     | 0.76    |
| Random Forest Classifier   | 0.82     | 0.76    |
| Gradient Boosting Classifier | 0.82   | 0.76    |

All models achieved similar performance. Gradient Boosting was selected for its balance of accuracy, robustness, and interpretability.

---

## 📈 Model Evaluation

- **Confusion Matrix** for class-wise accuracy
- **ROC AUC Score** to assess ranking performance
- **Classification Report** for precision, recall, F1-score

<details>
<summary>📉 Metrics Snapshot</summary>

```text
Accuracy: 0.82
ROC AUC Score: 0.76
Precision: 0.58
Recall: 0.29
F1-score: 0.39
