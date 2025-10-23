
# Student Data Analysis and Prediction

## Project Overview
This project performs **data preprocessing, clustering, regression modeling, and visualization** on a student dataset. It demonstrates a complete workflow for exploratory data analysis (EDA), handling skewed data, outlier treatment, feature scaling, clustering, and predictive modeling using machine learning algorithms.

### STUDENT PERFORMANCE PREDICTION APP:
https://clusteringtask-1.streamlit.app/

## Features
1. **Data Preprocessing**
   - Handles categorical variables using Label Encoding.
   - Corrects skewed numeric features with logarithmic transformation.
   - Treats outliers using the Interquartile Range (IQR) method.

2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap visualization for feature relationships.
   - Statistical overview of the dataset.

3. **Clustering**
   - Implements **KMeans**, **Agglomerative Clustering**, and **DBSCAN**.
   - Evaluates clustering performance using the **Silhouette score**.
   - Visualizes clusters in 2D space using **PCA**.

4. **Regression Modeling**
   - Uses **Random Forest Regressor** to predict the target variable.
   - Splits data into training and testing sets (80/20).
   - Evaluates model performance with **R² Score** and **RMSE**.

5. **Model & Scaler Saving**
   - Saves the trained model (`model.pkl`) and the scaler (`scaler.pkl`) for deployment or future predictions.

## Visualization:

### Skeness:
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/94cb315d-0faa-42a9-9d77-e0c9b8322d89" />


### Normal:
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/173b4290-1b85-477f-aed5-be7dcdeefef8" />


### Outilers(Before)
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/ef776af8-a39f-4cd7-a30c-72dafc8766ec" />


### Outliers(After)
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/3fed2ca3-f434-4f6d-b70b-0c8d4002c5c9" />



### Correlation :
<img width="776" height="682" alt="image" src="https://github.com/user-attachments/assets/ad2dfbd0-a35e-4a40-8170-1ad21602bbdc" />


### Clustering:

#### KMeans:
<img width="534" height="470" alt="image" src="https://github.com/user-attachments/assets/3bfcd62f-0536-451e-be1d-b28faebbf183" />


#### Agglomerative:
<img width="534" height="470" alt="image" src="https://github.com/user-attachments/assets/0310117f-8a77-4744-93d0-86ac56515770" />


#### DBSCAN :
<img width="534" height="470" alt="image" src="https://github.com/user-attachments/assets/5114b923-97f7-4d8b-8499-bf4033fcf51a" />


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-data-analysis.git
   cd student-data-analysis
