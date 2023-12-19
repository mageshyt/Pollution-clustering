import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

df=pd.read_csv('./data/train_cleaned.csv')


X=df.drop(['Name of station','DBU Class'],axis=1)

scaler=StandardScaler()

# standardize the data
X=scaler.fit_transform(X)

kmeans=KMeans(n_clusters=3,random_state=42)

kmeans.fit(X)


df['cluster-3']=kmeans.fit_predict(X)

px.scatter(df,x='D.O mg/L',y='pH',color='cluster-3')


## if we find to find the clustring for specific columns 

selected=['Nitrate mg/L','Nitrite mg/L'] # columns to be selected

for k in range(1,6):
    kmeans=KMeans(n_clusters=k,random_state=0,verbose=0)

    kmeans.fit(df[selected])

    df[f'cluster-{k}']=kmeans.fit_predict(df[selected])


fig,ax=plt.subplots(ncols=5,nrows=1,figsize=(20,5))
for i ,ax in enumerate(fig.axes,start=1):
    sns.scatterplot(x='D.O mg/L',y='pH',hue=f'cluster-{i}',data=df,ax=ax)


plt.tight_layout()