import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.table import Table
from sample_survey import survey
from music_recommender import recommend

# Init
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'], client_secret=os.environ['SPOTIFY_CLIENT_SECRET']))
data = pd.read_csv(os.environ['DATASET_PATH'], encoding = "utf-8")

def main_menu():
    console = Console()
    console.print("[bold green]Welcome to the ECS170 Music Recommender Project![/bold green]")
    console.print("------------------------------------------------")
    
    while True:
        menu_completer = WordCompleter(['Start Survey', 'Exit'])
        choice = prompt("Please select an option: ", completer=menu_completer)

        if 'Start Survey' in choice:
            console.print("[green]\nStarting the survey...[/green]")
            user_profile = survey()
            console.print("[green]\nGenerating recommendations...[/green]")
            recommend(data, user_profile)  # I think this prints out the recommendations or it might be print_recs()
        
        elif 'Exit' in choice:
            console.print("[bold green]\nThank you for using our Spotify based recommender![/bold green]")
            break

if __name__ == "__main__":
    main_menu()
