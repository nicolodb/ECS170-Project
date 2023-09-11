
metrics = ['danceability','energy','loudness','speechiness','acousticness',
               'instrumentalness','liveness','tempo','valence']

pop_genres = ['alternative','blues','classical','dubstep','country','edm',
             'electronic','hip-hop','house','indie-pop','indie','j-pop','jazz',
             'k-pop','latin','latino','metal','pop','punk-rock','r-n-b','reggae','rock','soul',
             'techno']

definitions = {'danceability':'how danceable a song is','energy':'intenstity of the song',
           'loudness': 'loudness of a song in decibels (dB)', 'speechiness':'detects the presence of spoken words (e.g an audio book would have high speechiness,'
           ' while a song would have a mid to low speechiness', 'acousticness': 'whether the song is acoustic or not', 'liveness': 'detects the prescence of an audience in the recording',
           'valence': 'describes the musical positveness conveyed by a song (e.g. high valence sounds more positive while low valence sounds more negative)', 'tempo': 'estimated beats per minutes (BPM)'
            ,'instrumentalness':'predicts whether the song contains no vocals'}

columns = ['track_id','artists','album_name','track_name','popularity','duration_ms',
           'explicit','time_signature','track_genre']

genre_index = ['danceability','energy','loudness','speechiness','acousticness',
               'instrumentalness','liveness','tempo','valence', 'track_genre']


likert_mapping = {"danceability": [0,.25,5,.75,1],"energy":[0,.25,.5,.75,1],"loudness":[-49.531,-20,-5,0,4.532],
                  "speechiness":[0,.193,.386,.772,.965], "acousticness":[0,.25,.5,.75,1], "instrumentalness":[0,.25,.5,.75,1],
                  "liveness": [0,.25,.5,.75,1], "valence": [0,.25,.5,.75,1], "tempo": [0,30,60.8,121,243.372]}

feature_weights = {'danceability': .1, 'energy': 0.3, 'loudness': 0.25, 'speechiness':0,
                   'acousticness':.15,'instrumentalness':.1, 'liveness':0,
                   'tempo':0,'valence':.1}
                   
