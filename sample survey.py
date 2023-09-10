import random
import pandas as pd
import os
import spotipy
import credentials
import info
from spotipy.oauth2 import SpotifyClientCredentials

# make sure to change this to your file path
data = pd.read_csv(os.environ['DATASET_PATH'], encoding = "utf-8")
columns = data.columns
# you can get your own client id and client secret by creating an app on spotify for developers
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'],client_secret=os.environ['SPOTIFY_CLIENT_SECRET']))


def error(number):
    if number == 1:
        print("\nInvalid answer. Please try again!\n")
    elif number == 2:
        print("\nYou already entered this song!\n")
    elif number == 3:
        print("\nNo matching tracks found. Please try again!\n")
        
def question_text(number):
    if number == 1:
        return "Would you like to filter out songs with explicit content?"
    elif number == 2:
        return "There's a lot of different genres! Would you like to see only the most popular ones?"
    elif number == 3:
        return ("Would you like to specify how much of each audio feature you would like? (e.g. loudness, tempo, etc). It's not required,"
               " but it is highly recommended for a better recommendation!")
    elif number == 4:
        return "What's your top five favorite songs?"
    elif number == 5:
        return "Pick a genre from the list!"
        

def responses():
    responses = ["Perfect!", "Okay!", "Great choice!",
                 "Awesome!", "Great!", "Good!", "Amazing!",
                 "Thank you for answering!", "Nice answer!",
                 "Good choice!", "Of course!", "Alright!",
                 "I'll get right on that!", "You have great taste!"]
    print("\n"+random.choice(responses))

def usage(number,answer_options, q):
    # how to answer the question + the options 
    if number == 1:
        print("\nPlease answer the following question by choosing one of the following options:\n")
        print(question_text(q))
        print("* ",end='')
        print(*answer_options,sep='\n* ')
        print("\n")
    # how to answer a likert scale
    elif number == 2:
        print(*answer_options,sep="     ")
        print("none     ",
              "neutral     ",
              "alot")
    # how to enter a song
    elif number == 3:
        print("\nPlease enter your song using the following format: [song name],[artist]")
        print("\t-> example: Billie Jean, Michael Jackson")
        
def check_method(type, answer, answer_options):
     if type == 'text':
           check = ((answer.upper() in answer_options) or (answer.lower() in answer_options))
     elif answer.isnumeric():
           check = (int(answer) in answer_options)
     else:
         return 0
     return check

def question(answer_options,type,u_index,q_index):
   looped = False
   while True:
       if looped == False:
           usage(u_index,answer_options,q_index)
       answer = input("answer:\t")
       check = check_method(type,answer,answer_options)
       if check:
           responses()
           print("------------")
           return answer.upper()
       else:
           error(1)
           looped = True
        
def likert_scale():
    answers = []
    scale = [1,2,3,4,5]
    for metric in info.metrics:
                print("On a scale of 1-5, how much",metric,"do you want?")
                print("\ndefinition:",info.definitions[metric],"\n")
                answer = question(scale,'n/a',2,0)
                answers.append(answer)
    return answers

def validate_song():
    while True:
        usage(3,0,0)
        song = input("answer:")
        song = song.title().split(',')
        trimmed = [s.strip() for s in song]
        try:
            # if there's a row that has matches both the song name and artist, 'row' will be initalized to that row 
            row = data[(data['track_name'] == trimmed[0]) & (data['artists'].apply(lambda x: trimmed[1].strip() in str(x)))]
            # if the song exists in spotify's catalog, 'results' will be initalized to it 
            results = sp.search(q = 'tracks: {}, artist{}, year{}'.format(trimmed[0],trimmed[1],trimmed[-1]),limit=1)
            return [trimmed,row,results]
        except:
            error(1)

def find_song():
    count = 0
    song_list = []
    while count != 5:
        temp = validate_song()
        song = temp[0]
        row = temp[1]
        results = temp[-1]
        # checks if the song was able to be found 
        if not row.empty or results['tracks']['items'] != []:
            if song not in song_list:
                print("\n")
                responses()
                print("------------")
                song_list.append(song)
                count+=1
            else:
                error(2)
        else:
            error(3)
    
    return song_list
      
    
def survey():
    """
    these are just sample questions.
    we can change or add more questions as needed. i just wanted to demonstrate how the quiz and question format would work! 
    """
    # creates a list containing every genre in the dataset 
    genres = data['track_genre'].unique().tolist()
    yes_no = ['Y','N']
    # question #1----------
    explicit = question(yes_no,'text',1,1)
    # question #2----------
    filter_genres = question(yes_no,'text',1,2)
    if filter_genres == 'Y': # only the most popular genres 
        curr_genres = question(info.pop_genres,'text',1,5)
    else:
        curr_genres = question(genres,'text',1,5) # all genres in the dataset
    # question 3-----------
    metric = question(yes_no,'text',1,3)
    if metric == 'Y':
        metrics = likert_scale()
    else:
        metrics = None
    # question 4-----------
    question_text(4)
    song_list = find_song()
    
    return [explicit,curr_genres,metrics,song_list]
