# 🛍️ Customer Retention Prediction Model

## 📖 Overview

The **Customer Retention Prediction Model** focuses on identifying patterns in customer behavior and predicting their likelihood of retention or churn. By analyzing detailed transaction data, demographic information, and tenure with the organization, this model provides actionable insights to enhance customer loyalty.

---

## 📊 Data Description

The dataset consists of the following key attributes:

| **Column Name**                  | **Description**                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------|
| **CIF**                          | Unique Customer Identifier.                                                    |
| **CUS_DOB**                      | Customer's Date of Birth.                                                      |
| **AGE**                          | Age of the customer (calculated from `CUS_DOB`).                               |
| **CUS_Month_Income**             | Monthly income of the customer.                                                |
| **CUS_Gender**                   | Gender of the customer (`Male`, `Female`, or `Other`).                         |
| **CUS_Marital_Status**           | Marital status of the customer (`Single`, `Married`, etc.).                    |
| **CUS_Customer_Since**           | Date when the customer joined the organization.                                |
| **YEARS_WITH_US**                | Total years the customer has been with the organization (calculated).          |
| **# total debit transactions for S1** | Total debit transactions made in Season 1.                                    |
| **# total debit transactions for S2** | Total debit transactions made in Season 2.                                    |
| **# total debit transactions for S3** | Total debit transactions made in Season 3.                                    |
| **total debit amount for S1**    | Total debit amount in Season 1.                                                |
| **total debit amount for S2**    | Total debit amount in Season 2.                                                |
| **total debit amount for S3**    | Total debit amount in Season 3.                                                |
| **# total credit transactions for S1** | Total credit transactions made in Season 1.                                  |
| **# total credit transactions for S2** | Total credit transactions made in Season 2.                                  |
| **# total credit transactions for S3** | Total credit transactions made in Season 3.                                  |
| **total credit amount for S1**   | Total credit amount in Season 1.                                               |
| **total credit amount for S2**   | Total credit amount in Season 2.                                               |
| **total credit amount for S3**   | Total credit amount in Season 3.                                               |
| **total debit amount**           | Total debit amount across all seasons.                                         |
| **total debit transactions**     | Total debit transactions across all seasons.                                   |
| **total credit amount**          | Total credit amount across all seasons.                                        |
| **total credit transactions**    | Total credit transactions across all seasons.                                  |
| **total transactions**           | Total number of transactions (debit + credit).                                 |
| **CUS_Target**                   | Target variable indicating retention or churn (`0 = Retention`, `1 = Churn`).  |
| **TAR_Desc**                     | Description of the target variable.                                            |
| **Status**                       | Additional status information (`Active`, `Inactive`, etc.).                    |

---

## 🧪 How the Model Works

1. **Data Preprocessing**
   - Extract key features like `AGE`, `YEARS_WITH_US`, and aggregated transaction metrics.
   - Convert dates to relevant formats and handle missing or invalid data.
   - Normalize income, transaction counts, and transaction amounts for effective modeling.

2. **Exploratory Data Analysis (EDA)**
   - Analyze the distribution of customer demographics and transaction patterns.
   - Identify correlations between features like `AGE`, `YEARS_WITH_US`, and `CUS_Target`.

3. **Feature Engineering**
   - Combine seasonal data into comprehensive metrics (e.g., total transactions, total amounts).
   - Create categorical features for customer segmentation based on demographics and behavior.

4. **Model Training**
   - Train a classification model (e.g., Random Forest, XGBoost) to predict `CUS_Target`.

5. **Evaluation**
   - Evaluate performance using metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 📈 Key Features

- **Transaction Insights**: Gain detailed insights into customer spending patterns.
- **Customer Demographics**: Understand the role of age, income, and marital status in retention.
- **Predictive Modeling**: Forecast churn probabilities and identify high-risk customers.
- **Visualization**: Use graphs and plots to explore trends in customer data.

---

## 🚀 Future Work

- **Enhanced Segmentation**: Introduce clustering techniques to identify distinct customer groups.
- **Time Series Analysis**: Incorporate sequential transaction data for deeper insights.
- **Personalized Recommendations**: Use model outputs to suggest targeted offers and services.
- **Deployment**: Build an interactive dashboard for business teams to monitor customer retention in real-time.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
