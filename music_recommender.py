import credentials
import info 
import os
import spotipy
import pandas as pd
import numpy as np
import random
import time
import sys
import warnings
from kmean import song_cluster, genre_cluster
from sample_survey import survey
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.spatial.distance  import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""
this is pretty much finished! it's still a little inconsistent (sometimes it will give zero recs), but for our purposes, i think it's good enough.
also, more error handling needs to be added and maybe also more filtering/finetuning.
feel free to add new features or improve some of the functions! :)

note:
    for some reason a lot of the recommendation tends to be genres such as dubstep and metal, idk if that's a problem with my implementation/filtering or the dataset itself
    -> so naturally, the recommendations are better when the user has a song list of songs in those genres 
"""

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'],client_secret=os.environ['SPOTIFY_CLIENT_SECRET']))
pd.set_option('display.max_columns',None)

def error(num):
    if num == 1:
        print("This song does not exist in our dataset or in Spotify's catalog!")
    elif num == 2:
        print("We do not have enough songs to make a recommendation. Please retake the survey.")

def loading_animation():
    chars = "/-\|"
    for _ in range(20):
        for char in chars:
            sys.stdout.write(f"\rLoading... {char}   ")
            sys.stdout.flush()
            time.sleep(0.1)
            
def extract_features(name, artist):
    """
    if a song is not in the dataset, we search for it in spotify's catalog and
    extract it's audio features
    """
    song_data = {}
    # searches the spotify catalog for the song
    # idk if i'm accessing it correctly so that's a potential problem
    results = sp.search(q= 'tracks: "{}" artist: "{}"'.format(name,artist), limit=1)
    # not in the catalog
    if results['tracks']['items'] == []:
        return None
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['track_name'] = [results['name']]
    song_data['artists'] = [results['artists']]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['popularity'] = [results['popularity']]
    song_data['duration_ms'] = [results['duration_ms']]
    for key, value in audio_features.items():
        song_data[key] = value
    
    return pd.DataFrame(song_data)
    
def get_data(name, artist, dataset):
    """
    checks if a song in the user's list is in our dataset. if not, we get the song data
    from the spotify catalog
    """
    try:
        # check if it's in the dataset
        song_data = dataset[(dataset['track_name'] == name) &
                            (dataset['artists'] == artist)]
        # in dataset
        if not song_data.empty:
            return song_data
        # if it's not in the dataset, check the spotify catalog
        data = extract_features(name, artist)
        # in the spotify catalog  
        if data is not None:
            return data
        else:
            raise ValueError("Song cannot be found")
            exit()
    
    except Exception as e:
        print(f"Error: {e}")
        return None       

def mean_vector(song_list, dataset, columns):
    """
    gets the mean vector for the user's list of songs
    """
    vectors = []
    # keep track of songs we can't find. can't find 3+ -> user takes the survey again
    # hasn't been implemented yet, but not incredibly important to do so 
    count = 0
    for song in song_list:
        data = get_data(song[0],song[1],dataset)
        # song data cannot be found 
        if data is None or data.empty:
            error(1)
            count+=1
            continue
        # forces to the user to retake the survey (not implemented yet)
        if count > 2:
            error(2)
        song_vector = data[columns].values
        vectors.append(song_vector)
    matrix = np.array(list(vectors))
    return np.mean(matrix, axis=0)

def scaled(data,scaler,columns=None):
    """
    standardize the data
    """
    if isinstance(data,list):
        data_2d = np.reshape(data, (1,-1))
        scaled_data = scaler.transform(data_2d)

    # assumes it's a dataframe 
    else:
        if columns != None:
            scaled_data = scaler.transform(data[columns])
        #else:
            #scaled_data = scaler.transform(data)
    return scaled_data

def assign_user(pipeline,scaler, user_profile):
    """
    assigns the user to a cluster based on their survey answers
    """
    metrics = user_profile[2]
    # default values for missing features
    # probably a better way to do this 
    avg = sum(int(n) for n in metrics)/len(metrics)
    metrics[:0] = [avg]
    metrics.insert(3,avg)
    metrics.insert(4,avg)
    metrics.append(avg)
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


def popularity(num,cs):
    """
    sorts by popularity then returns the num number of songs
    """
    indices = np.argsort(cs[0])[::-1]
    top_num = indices[:num]
    return top_num

def cosine_sim(first,second):
    """
    calculates the cosine similarity between two arrays 
    """
    # convert to a 2D array 
    first= np.array(first)
    second = np.array(second)
    cs = cosine_similarity(first, second)
    return cs

def get_genre_cluster(data,pipeline,subset,looped=False):
    """
    finds the genre's cluster and returns a list of all the other genres in the cluster
    """
    # gets the scaler used
    scaler = pipeline.named_steps['scaler']
    # standardizes the data
    if isinstance(data,pd.DataFrame):
        # only numeric data
        num = data.select_dtypes(np.number)
        scaled_data = scaled(num,scaler,info.metrics)
    else:
        scaled_data = np.array(data).reshape(1,-1)
        
    # finds the song's genre cluster
    cluster = pipeline.predict(scaled_data)[0]
    # finds the genres in the cluster
    cluster_genres = subset[subset['cluster_label'] == cluster]
    # list of all the genres
    genres = [genre for genre in cluster_genres['track_genre']]
    """
    only adds the nearby cluster once. if the user's song list contains a variety of genres, we
    might end up adding every cluster, making hard to make a tailored recommendation.
    """
    if looped == False:
        # finds the genres in a nearby cluster for diversity
        if cluster != 0:
            nearby_genres = subset[subset['cluster_label'] == cluster-1]
            genres.extend([genre for genre in nearby_genres['track_genre']])
        elif cluster != 8:
            more_genres = subset[subset['cluster_label'] == cluster+1]
            genres.extend([genre for genre in more_genres['track_genre']])

    return genres 

def genre_filtering(genres,songs):
    """
    filters the song to only include songs whose genres are in the cluster
    """
    filtered = []
    # loops through the songs
    for index, row in songs.iterrows():
        # checks if the song's genre is in the list of acceptable genres 
        if row['track_genre'] in genres:
            name = row['track_name']
            artist = row['artists']
            popularity = row['popularity']
            explicit = row['explicit']
            filtered.append([name,artist,explicit, popularity])
    return filtered

def pop_filtering(num,cs,cluster_songs,genres=None):
    """
    filters the recommendations to only include songs whose popularity
    satisfies an arbitrary popularity threshold
     - fyi iirc the average popularity is ~33
    """
    # gets the num top songs
    top = popularity(num,cs)
    # gets the top num songs info 
    top_songs = cluster_songs.iloc[top]
    # filters out the songs that are not in the genre cluster
    if genres != None:
        valid_recs = genre_filtering(genres,top_songs)
    # gets the songs that satisfy the popularity threshold 
    filtered_songs = [item for item in valid_recs if item[-1] >= 60]
    # gets rid of duplicates
    filtered_songs = [song for index, song in enumerate(filtered_songs) if song not in filtered_songs[:index]]
    return filtered_songs

def make_recs(data,scaler,columns,g_pipeline,subset,song_data,num,s_pipeline,looped=False):
    """
    the general process of getting the recs regardless of the method:
    1. standardizes the data
    2. find the clusters
    3. find all the songs or genres in the same cluster
    5. calculate the cosine similarity
    6. filter
    """
    # standardizes the data
    scaled_data = scaled(data,scaler,columns)
    # get a list of all the genres in the cluster
    genres = get_genre_cluster(data,g_pipeline,subset,looped)
    # finds the song's cluster
    cluster_label = s_pipeline.predict(scaled_data)[0]
    # finds the songs in the same cluster
    cluster_songs = song_data[song_data['cluster_label']==cluster_label]
    # standarizes the data
    scaled_similar = scaled(cluster_songs,scaler,columns)
    # calculates cosine similarity 
    cs = cosine_sim(scaled_data,scaled_similar)
    # filters songs by popularity and genres
    filtered_songs = pop_filtering(num,cs,cluster_songs,genres)
    return filtered_songs

def similar_songs(song_list,song_data,s_pipeline,scaler,columns,g_pipeline, subset):
    """
    for every song in the song list, it calculates the cosine similarity between the songs and
    the songs in its cluster. afterwards, the songs with the higest similarities are
    filtered by popularity and acceptable genres (i.e. the genres in the song's genres list and a nearby cluster)
        -> gives decent recs! :)
    """
    recs = []
    looped = False
    for song in song_list:
        data = get_data(song[0],song[1],song_data)
        # skips to the next song if song data cannot be found 
        if data is None or data.empty:
            continue
        # gets the song recommendation
        filtered_songs = make_recs(data,scaler,columns,g_pipeline,
                                  subset,song_data,800,s_pipeline,looped)
        recs.extend(filtered_songs)
        looped = True
    return recs

def user_pref_songs(profile,song_data,pipeline,scaler,columns,g_pipeline,subset):
    """
    assigns the user to a cluster based on their audio preferences. then calculates the cosine similarity
    between the user and the song in their cluster.
        -> decent recs
        -> good for diversifying the recs 
    """
    pref = profile[2]
    genres = get_genre_cluster(pref,g_pipeline,subset)
    # assigns the user to a song cluster
    user_cluster = assign_user(pipeline,scaler,profile)
    # finds all the songs in the user's cluster
    cluster_songs = song_data[song_data['cluster_label'] == user_cluster]
    # standardizes the data
    scaled_profile = scaled(profile[2],scaler,columns)
    scaled_songs = scaled(cluster_songs,scaler,columns)
    # calculates the cosine similarity between the user's audio preference and the songs in the cluster
    cs = cosine_sim(scaled_profile,scaled_songs)
    # filter songs by popularity
    filtered_songs = pop_filtering(800,cs,cluster_songs,genres)
    return filtered_songs

def vector_recs(center,song_data,profile,pipeline,scaler,g_pipeline,subset,columns):
    """
    follows the same format as the other methods. the only difference is we calculate the cosine
    similarity of the center/mean vector (a representative of all the audio features for the songs
    in the song list)
        -> similar recs to the similar_songs()
        -> may can omit or alter this more? 
    """
    # converts into a dataframe 
    df_center = pd.DataFrame(center,columns=columns)
    filter_songs = make_recs(df_center,scaler,columns,g_pipeline,subset,song_data,800,pipeline)
    return filter_songs 


def filtered(answer,song):
    """
    if the user answer 'y' to the question about filtering out
    explict songs, then we remove it from the recommendations list.
        -> maybe filter can filter out other stuff
    """
    # skip over the song 
    if answer == 'Y' and song[0][2] == True:
        return 1
    return 0


def pick_random(num,songs,curr_recs,explicit):
    """"
    randomly picks num songs
    """
    recs = []
    # not enough songs
    if len(songs) < num:
        if len(songs) == 0:
            return recs
        return songs
            
    while len(recs) < num:
        song = random.choice(songs)
        if filtered(explicit,song) == 1:
            continue 
        if song not in curr_recs and song not in recs:
                recs.append(song)
                if len(recs) == num:
                    break 
    return recs

def print_recs(recs):
    """
    prints the track name and artist(s)
    """
    if recs == []:
        print("\nSorry! We couldn't recommend any songs. :(\n")
    for rec in recs:
        try:
            print(f"Name: {rec[0]}, Artist(s): {rec[1]}")
        except:
            for info in rec:
                print(f"Name: {info[0]}, Artist(s): {info[1]}")

def get_recs(ss,up,mv,profile):
    loading_animation()
    # list of songs already being recommended 
    recs = []
    ex = profile[0]
    method_ss = pick_random(6,ss,recs,ex)
    recs.append(method_ss)
    method_up = pick_random(2,up,recs,ex)
    recs.append(method_up)
    method_mv = pick_random(2,mv,recs,ex)
    print("\nHere is your recommendations!\n")
    print("Based on your song list:\n")
    print_recs(method_ss)
    print("\nBased on your audio preferences:\n")
    print_recs(method_up)
    print("\nBased on the mean vector of your song list:\n")
    print_recs(method_mv)
    
def recommend(dataset,profile, n=10):
    """
    provides recommendations with three different methods:
    1. find similar songs to the user's song list
    2. assigning the user to a cluster first based on their audio features preferences 
    3. clustering the mean vector
    """
    print("\nMaking your recommendations...\n")
    loading_animation()
    # supresses all the clustering and scaling warnings
    original_filters = warnings.filters[:]
    warnings.filterwarnings("ignore")
    # k mean clustering 
    pipeline, cluster_labels,song_data = song_cluster()
    # only numeric values 
    X = dataset.select_dtypes(np.number)
    features = X.iloc[:,2:15]
    columns = features.columns.tolist()
    # genre clustering
    g_pipeline, subset = genre_cluster()
    # calculates the mean vector 
    center = mean_vector(profile[-1], dataset,columns)
    scaler = pipeline.named_steps['scaler']
    # method one
    ss = similar_songs(profile[-1],song_data,pipeline,scaler,columns, g_pipeline,subset)
    # method two 
    up = user_pref_songs(profile,song_data,pipeline,scaler,columns,g_pipeline,subset)
    # method three 
    mv = vector_recs(center,song_data,profile,pipeline,scaler,g_pipeline,subset,columns)
    warnings.filters = original_filters
    get_recs(ss,up,mv,profile)
    
def run():
    data = pd.read_csv(os.environ['DATASET_PATH'], encoding = "utf-8")
    print("\nPlease take this survey so we can get a better idea of your music preferences!\n")
    print("----------------------------------------")
    user_profile = survey()
    recommend(data,user_profile)
    
run()
    
            

            
