import credentials
import info 
import os
import spotipy
import pandas as pd
import numpy as np
from kmean import song_cluster
from sample_survey import survey
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.spatial.distance  import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'],client_secret=os.environ['SPOTIFY_CLIENT_SECRET']))

def error(num):
    if num == 1:
        print("This song does not exist in our dataset or in Spotify's catalog!")
    elif num == 2:
        print("We do not have enough songs to make a recommendation. Please retake the survey.")
        
def extract_features(name, artist):
    song_data = {}
    # searches the spotify catalog for the song
    # idk if i'm accessing it correctly so that's a potential problem
    results  = sp.search(q= 'track: {} artist: {}'.format(name,artist), limit=1)
    # not in the catalog 
    if results['tracks']['items'] == []:
        return None
    
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['name'] = [name]
    song_data['artist'] = [artist]
    song_data['explicit'] = [int(results['explicit'])]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)
    
def get_data(song, dataset):
    """
    checks if a song in the user's list is in our dataset. if not, we get the song data
    from the spotify catalog
    """
    try:
        song_data = dataset[(dataset['track_name'] == song[0]) &
                            (dataset['artists'] == song[1])]
        if song_data.empty:
            raise ValueError("Not in dataset")
        return song_data
    
    except:
        return extract_features(song[0], song[1])


def mean_vector(song_list, dataset, columns):
    """
    gets the mean vector for the user's list of songs
    """
    vectors = []
    # keep track of songs we can't find. can't find 3+ -> user takes the survey again
    # hasn't been implemented yet, but not incredibly important to do so 
    count = 0
    for song in song_list:
        data = get_data(song,dataset)
        if data is None:
            error(1)
            count+=1
            continue 
        if count > 2:
            error(2)
        song_vector = data[columns].values
        vectors.append(song_vector)
    matrix = np.array(list(vectors))
    return np.mean(matrix, axis=0)

def assign_user(pipeline,scaler, user_profile):
    """
    assign's the user to a cluster based on their survey answers
    -> not really helpful since we don't have a dataset of profiles
    """
    metrics = user_profile[2]
    # default values for missing features
    avg = sum(metrics)/len(metrics)
    metrics[:0] = [avg]
    metrics.insert(3,0)
    metrics.insert(4,avg)
    metrics.append(1)
    # standarizes their metric answers and reshapes it to a 2D array 
    scaled_profile = scaler.transform(np.array(metrics).reshape(1,-1))
    # find the user's cluster
    cluster_label = pipeline.predict(scaled_profile)[0]
    return cluster_label

def map_likert(metrics):
    """
    maps the values on the 5 point likert scale to actual values from the data set
    """
    mapped = []
    count = 0
    for metric in info.metrics:
        scale = info.likert_mapping[metric]
        value = scale[int(metrics[count])-1]
        count+=1
        mapped.append(value)
    return mapped

def user_pref_songs(profile,song_data):
    """
    does not give great recommendations
    -> probably an error on my part :(
    -> maybe need to account for popularity
    """
    cs = []
    temp = profile[2]
    metrics = map_likert(temp)
    # reshape to a 2D array
    metrics = np.array(metrics).reshape(1,-1)
    # calculates cosine similarity
    for i, row in song_data.iterrows():
        song_metrics = np.array(row[info.metrics]).reshape(1,-1)
        score = cosine_similarity(metrics,song_metrics)
        cs.append(score)
    cs = np.array(cs).flatten()
    # creates a column for cosine_similarity
    song_data['cosine_similarity'] = cs
    return song_data

def print_similar(similar):
    for song in similar:
        name, artist = song
        print(f"Song: {name}, Artist: {artist}")
        
def similar_songs(song_list,song_data,pipeline,scaler,columns):
    """"
    gives somewhat okay recommendations but could be way better
    -> again, probably a problem with implementation
    -> some of the recommendations seem random 
    """
    similar = []
    for song in song_list:
        data = get_data(song[0],song[1])
        # standardizes the data
        scaled_data = scaler.transform(data[columns])
        # finds the song's cluster
        cluster_label = pipeline.predict(scaled_data)[0]
        # finds the songs in the same cluster
        cluster_songs = song_data[song_data['cluster_label']==cluster_label]
        similar.extend([(name, artist) for name, artist in
                        zip(cluster_songs['track_name'], cluster_songs['artists'])])
    # removes duplicates
    similar = list(set(similar))
    #print_similar(similar)
    return similar

def filtered(profile,recs,song_data):
    """
    if the user answer 'y' to the question about filtering out
    explict songs, then we remove it from the recommendations list.
    maybe filter can filter out other stuff
    """
    pass
            
def vector_recs(center, song_data,profile):
    """
    needs to be fixed/finished
    """
    recs = []
    for song in song_data:
        features = np.array(song[info.metrics].valy)
        similarity = np.dot(center,features)
        recs.append((song['track_name'],similarity))
    # sort recommendations by descending similarity
    recs.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:5]

def print_pref(up):
    unique = set()
    top_ten = up.sort_values(by='cosine_similarity', ascending=False).head(30)
    for idx, row in top_ten.iterrows():
        unique.add((row['track_name'], row['artists'], row['cosine_similarity']))
    for song in unique:
        track_name, artists, cosine_similarity = song
        print(f"Song: {track_name}, Artist: {artists}, Cosine Similarity: {cosine_similarity}")
    
def get_recs(s_songs,up,profile,song_data):
    print("similar song clustering\n")
    print(*s_songs[0:10],sep="\n")
    print("\nTop 10 songs with the highest cosine similarity:\n")
    print_pref(up)
    
                   
def recommend(dataset,profile, n=10):
    # k mean clustering 
    pipeline, cluster_labels,song_data = song_cluster()
    # only numeric values 
    X = dataset.select_dtypes(np.number)
    features = X.iloc[:,2:15]
    columns = features.columns.tolist()
    # calculates the mean vector 
    center = mean_vector(profile[-1], dataset,columns)
    scaler = pipeline.named_steps['scaler']
    # different methods:
    # 1. find similar songs to the user's song list
    s_songs = similar_songs(profile[-1],song_data,pipeline,scaler,columns)
    # 2. calculates the cosine similarity between the profile and all the songs
    up = user_pref_songs(profile,song_data)
    # 3. calculates the cosine similarity between the mean vector and each song
        # -> need to implement this
        #mv = vector_recs(center,song_data,profile)
    
    get_recs(s_songs,up,profile,song_data)

def run():
    data = pd.read_csv(os.environ['DATASET_PATH'], encoding = "utf-8")
    user_profile = survey()
    recommend(data,user_profile)
    
run()
        
            
