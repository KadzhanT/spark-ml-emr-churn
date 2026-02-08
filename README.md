Bank Customer Churn Prediction using Spark ML on Amazon EMR
This repository contains a PySpark application designed to predict customer churn using the Bank Customer Churn Dataset. The project is implemented as an end-to-end Machine Learning Pipeline running on a distributed Amazon EMR cluster.
1. Project Objective
The goal of this lab is to demonstrate the scalability of Spark ML by:
Building a distributed data processing pipeline.
Performing feature engineering on a real-world dataset.
Comparing different machine learning models in a distributed environment.
Deploying and monitoring Spark jobs using YARN and Amazon EMR.
2. Dataset Description
The dataset used is the Bank Customer Churn Dataset from Kaggle.
Source: Kaggle Link
Target Variable: Exited (0 = Stayed, 1 = Churned)
Key Features:
CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary (Numerical)
Geography, Gender (Categorical)
HasCrCard, IsActiveMember (Binary)
3. Tech Stack
Platform: Amazon EMR (Elastic MapReduce)
Cluster Configuration: 1 Master Node, 2 Core Nodes (m4.large)
Processing Engine: Apache Spark 3.x (PySpark)
Storage: HDFS (Hadoop Distributed File System)
Cluster Manager: YARN
4. ML Pipeline Stages
The application implements the following Spark ML Pipeline:
Data Ingestion: Reading CSV data from HDFS.
StringIndexer: Converting categorical strings (Geography, Gender) into numerical indices.
OneHotEncoder: Converting numerical indices into binary vectors.
VectorAssembler: Combining all feature columns into a single feature vector.
StandardScaler: Normalizing features to ensure model stability.
Model Training: Training the classification models.
5. How to Run
Step 1: Upload Data to EMR
First, upload the dataset to the Master Node and then move it to HDFS:

# From your local machine
scp -i vockey.pem Churn_Modelling.csv hadoop@<EMR-MASTER-DNS>:/home/hadoop/

# On the EMR Master Node
hdfs dfs -mkdir -p /user/hadoop/churn_input
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/

Step 2: Submit the Spark Job
Run the following command on the Master Node to execute the pipeline:

spark-submit \
  --master yarn \
  --deploy-mode client \
  churn_pipeline.py
  
6. Experiment Results (Option C: Model Comparison)
For this lab, I conducted an experiment comparing two different classification algorithms: Logistic Regression and Random Forest.
Model	Accuracy	Observation
Logistic Regression	~81.2%	Faster to train, but assumes linear relationships.
Random Forest	~86.5%	Higher accuracy as it captures non-linear patterns in customer behavior.
Conclusion: The Random Forest model provided better predictive performance for churn detection in this dataset.
