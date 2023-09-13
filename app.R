# Comparing Artists and Comparing Genres
# Jaishree Ramamoorthi

# Link to working website: https://1c6efv-jramamoorthi.shinyapps.io/genre_artist_comparisons/

# Load the libraries
library(shiny)
library(shinythemes)
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(DT)  # Load the DT package

# Define the Kaggle dataset URL
kaggle_dataset_url <- "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download"

# Load the TIDY dataset into a data frame
music_data <- read_csv("tidy_spotify_data.csv")
str(music_data)

# Split songs with multiple artists into individual artists
music_data <- music_data %>%
  separate_rows(artists, sep = ";")

# Calculate average metrics for each artist
artist_avg_metrics <- music_data %>%
  group_by(artists) %>%
  summarize(
    avg_popularity = mean(popularity),
    avg_duration_ms = mean(duration_ms),
    avg_danceability = mean(danceability),
    avg_energy = mean(energy),
    avg_key = mean(key),
    avg_loudness = mean(loudness),
    avg_mode = mean(mode),
    avg_speechiness = mean(speechiness),
    avg_acousticness = mean(acousticness),
    avg_instrumentalness = mean(instrumentalness),
    avg_liveness = mean(liveness),
    avg_valence = mean(valence),
    avg_tempo = mean(tempo),
    avg_time_signature = mean(time_signature)
  )

# Calculate average metrics for each genre
genre_avg_metrics <- music_data %>%
  group_by(track_genre) %>%
  summarize(
    avg_popularity = mean(popularity),
    avg_duration_ms = mean(duration_ms),
    avg_danceability = mean(danceability),
    avg_energy = mean(energy),
    avg_key = mean(key),
    avg_loudness = mean(loudness),
    avg_mode = mean(mode),
    avg_speechiness = mean(speechiness),
    avg_acousticness = mean(acousticness),
    avg_instrumentalness = mean(instrumentalness),
    avg_liveness = mean(liveness),
    avg_valence = mean(valence),
    avg_tempo = mean(tempo),
    avg_time_signature = mean(time_signature)
  )

# Define the Shiny UI
ui <- fluidPage(
  theme = shinytheme("yeti"),  # Apply the "yeti" theme
  titlePanel(
    "Spotify Music Analysis"
  ),
  sidebarLayout(
    sidebarPanel(
      selectInput("comparison_type", "Select Comparison Type:",
                  choices = c("Compare Artists", "Compare Genres"),
                  selected = "Compare Artists"),  # Input for selecting comparison type
      
      width = 0.5
      
    ),
    mainPanel(
      h3("Dataset Description"),
      HTML(paste("This dataset contains information on Spotify tracks from various genres. Each track is associated with audio features and is provided in CSV format. The dataset can be used for building recommendation systems, classification tasks based on audio features and genres, and other data-driven applications.",
                 "<br><strong>Original Kaggle Dataset:</strong> <a href='", kaggle_dataset_url, "' target='_blank'>Link</a>")),
      
      h3("App Purpose"),
      HTML("This app allows users to interact with Spotify track data and compare artists and genres based on various audio characteristics. Use the dropdown to select whether you want to compare artists or genres, and explore average metrics for each. The data is organized for easy analysis.\n \n \n"),
      
      tabsetPanel(
        tabPanel("Average Metrics by Artist", DTOutput("artist_metrics")),  # Use DTOutput
        tabPanel("Average Metrics by Genre", DTOutput("genre_metrics"))    # Use DTOutput
      )
    )
  )
)

# Define the Shiny server logic
server <- function(input, output) {
  # Filter data based on user selection (artists or genres)
  filtered_data <- reactive({
    if (input$comparison_type == "Compare Artists") {
      return(artist_avg_metrics)
    } else {
      return(genre_avg_metrics)
    }
  })
  
  # Render the average metrics tables with DT::renderDT
  output$artist_metrics <- DT::renderDT({
    datatable(filtered_data(), options = list(order = list(1, 'asc')))  # Initial sorting by the first column in ascending order
  })
  
  output$genre_metrics <- DT::renderDT({
    datatable(filtered_data(), options = list(order = list(1, 'asc')))    # Initial sorting by the first column in ascending order
  })
}

# Run the Shiny app
shinyApp(ui, server)
