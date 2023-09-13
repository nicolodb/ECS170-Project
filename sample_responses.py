# sample responses for survey
import random
from info import metrics, pop_genres, likert_mapping

# create random list of genres
def random_genres():
    num_genres = random.randint(1, len(pop_genres)) # can change this if you want to specify a number of genres to generate
    return random.sample(pop_genres, num_genres)

# create random likert ratings
def random_ratings():
    ratings = {}
    for metric in metrics:
        # use likert mapping min and max to generate random values for each metric
        min, max = likert_mapping[metric][0], likert_mapping[metric][-1]
        random_rating = round(random.uniform(min, max), 3) # rounded to 3 decimal places
        ratings[metric] = str(random_rating)
    return ratings

num_songs = 1000 # can change if you want more or less for the random songs from spotify
# create random song list
def random_songs(num_songs):
    # sample songs from spotify dataset
    selected_songs = data.sample(n=num_songs)
    songs = selected_songs[['track_name', 'artists']].values.tolist()
    
    return songs

# randomize answers to sample survey
def sample_answer():
    sample_answer = {
        "explicit": random.choice(["Y", "N"]),
        "filter_genres": random.choice(["Y", "N"]),
        "curr_genres": random_genres(), # generate random genres for testing *can change*
        "metric": random.choice(["Y", "N"]),
        "metrics": random_ratings(),
        "song_list": random_songs()
    }
    return sample_answer