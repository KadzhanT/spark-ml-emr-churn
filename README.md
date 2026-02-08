# 📊 Customer Churn Prediction on Amazon EMR
> **Lab 6:** Distributed Computing with Apache Spark ML Pipeline

![Spark](https://img.shields.io/badge/Apache_Spark-FDEE21?style=for-the-badge&logo=apachespark&logoColor=black)
![AWS](https://img.shields.io/badge/AWS_EMR-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🎯 Project Objective
This project demonstrates an end-to-end **Distributed Machine Learning Pipeline** to predict bank customer churn. By using **Apache Spark** on an **Amazon EMR** cluster, we process data at scale and compare different classification models.

---

## 📂 Dataset Overview
The project uses the **Bank Customer Churn Dataset**.
*   **Target:** `Exited` (1 = Churn, 0 = Stayed)
*   **Features:** Credit Score, Geography, Gender, Age, Balance, etc.
*   **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

---

## 🏗️ System Architecture
*   **Cloud Platform:** AWS Academy Learner Lab
*   **Service:** Amazon EMR (Elastic MapReduce)
*   **Cluster Nodes:**
    *   `1 Primary (Master)`: m4.large
    *   `2 Core`: m4.large
*   **Storage:** HDFS (Hadoop Distributed File System)

---

## 🛠️ Pipeline Stages
The application implements a modular `pyspark.ml.Pipeline`:
1.  **Preprocessing:** `StringIndexer` & `OneHotEncoder` for categorical data.
2.  **Vectorization:** `VectorAssembler` to merge features.
3.  **Scaling:** `StandardScaler` for numerical feature normalization.
4.  **Modeling:** Training and comparing multiple classifiers.

---

## 🚀 How to Execute

### 1. Upload Data to HDFS
Connect to your EMR Master node and run:
```bash
# Create directory in HDFS
hdfs dfs -mkdir -p /user/hadoop/churn_input

# Move CSV from local Master node to HDFS
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/

Execute the Python script using the YARN cluster manager:

spark-submit \
  --master yarn \
  --deploy-mode client \
  churn_pipeline.py


🧪 Experiment Results (Option C)
I performed a Model Comparison experiment between a linear model and an ensemble model.
Metric	Logistic Regression	Random Forest
Accuracy	      81.2%	86.4%
Training Speed	Fast	Moderate
Complexity	    Low	High
Observation:
The Random Forest model outperformed Logistic Regression by ~5%. This is because Random Forest can capture non-linear relationships between features like Age, Balance, and NumOfProducts which are critical for predicting churn.
