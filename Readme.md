# Customer Segmentation Using RFM Analysis and Clustering for Customer Value Profiling

This project performs customer segmentation on an online retail dataset using **RFM (Recency, Frequency, Monetary) analysis** and **K-Means clustering** to identify distinct customer groups and derive actionable business insights.

---

## 🚀 Project Overview

Customer segmentation is essential for understanding customer behavior and optimizing marketing strategies.  
In this project, customers are grouped based on purchasing behavior using RFM metrics and unsupervised clustering techniques.

The results are interpreted into meaningful **customer personas** such as Champions, Loyal Customers, At-Risk, and Low-Value segments.

---

## 📊 Dataset

- **Source**: Online Retail II Dataset (UCI Machine Learning Repository)
- **Period**: December 2009 – December 2011
- **Records**: >1 million transactions
- **Customers**: 5,878 unique customers
- **Main Fields**: Invoice, InvoiceDate, Quantity, Price, Customer ID, Country

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Removal of missing Customer IDs
- Exclusion of credit notes (returned transactions)
- Removal of negative or zero quantity and price values
- Deduplication of transactions
- Revenue calculation (`Revenue = Quantity × Price`)
- Conversion of InvoiceDate to datetime format

---

## 📐 Feature Engineering (RFM)

For each customer, three features were calculated:

- **Recency**: Days since the last purchase
- **Frequency**: Number of unique transactions
- **Monetary**: Total spending amount

Due to highly skewed distributions, **MinMaxScaler** was applied to normalize RFM features before clustering.

---

## 🤖 Clustering Methodology

- **Algorithm**: K-Means Clustering
- **Cluster Evaluation**:
  - Elbow Method (SSE)
  - Silhouette Score
  - Davies-Bouldin Index (DBI)
- **Optimal Number of Clusters**: `k = 4`  
  Selected based on combined evaluation metrics and cluster stability.

---

## 🧠 Customer Segments (Results)

The clustering process resulted in four distinct customer groups:

- **Champions**: Highly recent, frequent, and high-spending customers
- **Loyal Customers**: Consistent buyers with moderate to high engagement
- **At-Risk Customers**: Previously active customers with declining activity
- **Low-Value Customers**: Infrequent and low-spending customers

These segments can be used to support targeted marketing and retention strategies.

---

## 📈 Visualization

Key visualizations include:

- Elbow, Silhouette, and DBI plots
- RFM distribution per cluster
- PCA 2D and 3D cluster visualization
- Cluster profile comparisons (mean & median)

---

## 🌐 Live Application

- **Web App (Streamlit)**:  
  https://rfmclustering.streamlit.app/

- **Notebook (Google Colab)**:  
  https://colab.research.google.com/drive/1lBDsIiarQv1Mz87wFf2V_ofKbZvlthGI?usp=sharing

---

## ⚙️ How to Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/HilmyFahrizal/Customer_Segmentation_RFM
cd CustomerSegmentationRFMClustering
```

### 2 Install Library requirments
```bash
pip install -r requirements.txt
```

### 2 Run IPYNB File
- go to Notebooks Folder and Run All file Code_Capstone_RFM.ipynb

### 3. Run Streamlit
```bash
streamlit run Streamlit/CapstoneRFM_Deploy.py
```