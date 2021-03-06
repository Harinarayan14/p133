from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
df  = pd.read_csv("gravity_data.csv")
X = df.iloc[:,[3,4]]
wcss=[]
for i in range(1,10):
  kmeans = KMeans(n_clusters=i,init="k-means++",random_state=0)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))   
sns.lineplot(range(1,10),wcss,marker="o", color = "red")
plt.title("Elbow Curve")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

fig = px.scatter(x=df["Mass"], y=df["Radius"], hover_data=[df["Star_name"]],color=df["gravity"]) 
fig.show()