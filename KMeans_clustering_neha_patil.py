#!/usr/bin/env python
# coding: utf-8

# # Prediction using Unsupervised ML

# ## Task 2: K Means Clustering
# 
# by Neha Sandeep Patil

# ### From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# In[6]:


#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


# In[7]:


#load iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head() 


# In[8]:


#now we have to calculate the optimum number of clusters for k means clustering
x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []   # Within cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    


# In[9]:


# Plotting the results onto a line graph 
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS') 
plt.show()


# From above graph , we can say that - as No. of clusters increases the WCSS decreases significantly(up to 2).But the elbow occurs when the WCSS doesn't decreases significantly although No. of clusters increases.
# So the No. of cluster= 3 .

# In[10]:


# apply k means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)# n_clusters is the No. of clusters
y_kmeans = kmeans.fit_predict(x)


# In[11]:


# Visualising the clusters 
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'orange', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'red', label = 'Centroids')

plt.legend()


# So thus we have plotted the K means visually .
