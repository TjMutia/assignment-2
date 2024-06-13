import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('OnlineRetail.csv',encoding="unicode_escape")
print(data.head())
print(data.shape)
print(data.info())
print(data.columns)
print(data.describe())
print(data.isnull().sum())
data_null=round(100*(data.isnull().sum())/len(data),2)
print(data_null)
#droping
#data=data.drop(['StockCode'],axis=1)

data['CustomerID']=data['CustomerID'].astype(str)
print(data)
#data preparation
#monetary
data['Amount']=data['Quantity']*data['UnitPrice']
print(data.info())
data_monitoring=data.groupby('CustomerID')['Amount'].sum()
print(data_monitoring)
print(data_monitoring.head())
data_monitoring=data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print(data_monitoring)
print(data_monitoring.head())
data_monitoring=data.groupby('Country')['Quantity'].sum()
#print(data_monitoring)

print(data_monitoring.head())
data_monitoring=data.groupby('Description')['InvoiceNo'].count()
print(data_monitoring)
print(data_monitoring.head())
data_monitoring=data_monitoring.reset_index()
data_monitoring.columns=['CustomerID', 'Index']

data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
print(data)

# Compute the maximum date to know the last transaction date
max_date = max(data['InvoiceDate'])
print(max_date)

min_date = min(data['InvoiceDate'])
print(min_date)

#total number of day Compute the difference between max date and transaction date
data['Diff'] = max_date - data['InvoiceDate']
print(data.head())

#max_date = max(data['InvoiceDate'])-30
from datetime import timedelta
diff_time = max_date - timedelta(days=30)
dtt = data[data["InvoiceDate"] >= diff_time]
print(diff_time)
hjvhj =dtt["Amount"].sum()
print(hjvhj)

total_Amount_sales=dtt['Amount'].count()
print(total_Amount_sales)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Define the input data for clustering
X = data[['Amount']]
#Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X,'Numberof clusters')
    wcss.append(kmeans.inertia_)
    
# Plotting the Elbow Method
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Choose the optimal k 
optimal_k = 3

# Apply K-Means clustering with the chosen k
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Visualize Clusters
plt.figure(figsize=(10, 20))
plt.scatter(data['Amount'], data['Quantity'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Amount')
plt.ylabel('Quantity')
plt.title('Customer Segmentation: Amount vs Quantity')
plt.colorbar(label='Cluster')
plt.show()

