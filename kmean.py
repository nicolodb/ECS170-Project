import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt 
import info
import kaleido
import os
import credentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# cleaned dataset
song_data = pd.read_csv(os.environ['DATASET_PATH'], encoding = "utf-8")
pd.set_option('display.max_columns',None)
#print(song_data.head())

def subset_genre():
    # averages each the audio features for each genre 
    genre_data = song_data.groupby('track_genre')[info.metrics].mean().reset_index()
    # normalizes the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(genre_data[info.metrics])
    # assigns the normalized features
    genre_data[info.metrics] = normalized_features
    return genre_data
    

def elbow_method(df,title):
    """
    elbow method: find the ideal number of groups to divide the data into
    one way:
         -> find the ideal value of k (k = # of clusters)
    def: inertia_ = sum of squared distance of samples to their closest cluster center
     -> float data type
    """
    sse = [] #sum of squared error
    data = df[(info.metrics)] # only numerical data
    for k in range(1,11):
        k_mean = KMeans(n_clusters = k, random_state = 2)
        k_mean.fit(data)
        sse.append(k_mean.inertia_)
    # graph
    sns.set_style("whitegrid")
    graph = sns.lineplot(x = range(1,11), y = sse)
    graph.set(xlabel = "Number of cluster (k)",
              ylabel = "Sum Squared Error",
              title = title)
    #plt.show()
    
def elbow_method_genre():
    #applies the elbow method on genre subset
    subset = subset_genre()
    file_name = "kmean_genre.png"
    elbow_method(subset,'genres')
    # saves the graph to your computer
    plt.savefig(os.environ['DOWNLOAD_PATH']+file_name)

def genre_graph(X, genre_subset):
    file_name = "k=8_genre (clean).png"
    # compresses the data into a 2D space so it's easier to visualize 
    tsne_pipeline = Pipeline([('scaler',StandardScaler()),('tsne',TSNE(n_components=2,verbose=0))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x','y'],data=genre_embedding)
    projection['track_genre'] = genre_subset['track_genre']
    projection['cluster'] = genre_subset['cluster']

    # visualizes the clusters in a 2D space
    fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x','y','cluster','track_genre'])
    fig.update_layout(title='genre cluster') # adds a title
    fig.write_image(os.environ['DOWNLOAD_PATH']+file_name)
    fig.show()

def genre_cluster():
    genre_subset = subset_genre()
    # scales the data and applies the k mean clustering algorithm on the scaled data 
    cluster_pipeline = Pipeline([('scaler', StandardScaler()),('kmeans',KMeans(n_clusters=8))])
    # selects columns with only numeric data types
    X = genre_subset.select_dtypes(np.number)
    cluster_pipeline.fit(X) # trains on data
    genre_subset.loc[:,'cluster_label'] = cluster_pipeline.predict(X) # makes predictions
    #genre_graph(X,genre_subset)
    return cluster_pipeline, genre_subset
    

def song_graph(X):
    file_name = "k=6_song (clean).png"
    # compresses the data
    pca_pipeline = Pipeline([('scaler', StandardScaler()),  ('PCA',PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x','y'],data=song_embedding)
    projection['title'] = song_data['track_name']
    projection['cluster'] = song_data['cluster_label']

    # visualizes the song cluster
    fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x','y','cluster','title'])
    fig.write_image(os.environ['DOWNLOAD_PATH']+file_name)
    fig.show()
    
def song_cluster():
    # divides into n clusters
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),('kmeans',KMeans(n_clusters=6,verbose=0))]
                                     ,verbose =True)
    # selects columns with only numeric data types
    X = song_data.select_dtypes(np.number)
    features = X.iloc[:,2:15]
    song_cluster_pipeline.fit(features)
    song_cluster_labels = song_cluster_pipeline.predict(features)
    song_data['cluster_label'] = song_cluster_labels
    #song_graph(X)
    return song_cluster_pipeline, song_cluster_labels, song_data


    

    
    
