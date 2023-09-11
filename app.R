# Comparing Artists and Comparing Genres
# Jaishree Ramamoorthi

# Load the libraries
library(shiny)
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(DT)  # Load the DT package

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
  theme = shinytheme("journal"),  # Apply the "flatly" theme
  titlePanel(
    "Spotify Music Analysis",
    div(
      style = "font-size: 14px; margin-top: 5px;",  # Style the description
      "Explore average metrics for artists and genres."
    )
  ),
  sidebarLayout(
    sidebarPanel(
      selectInput("comparison_type", "Select Comparison Type:",
                  choices = c("Compare Artists", "Compare Genres"),
                  selected = "Compare Artists"),  # Input for selecting comparison type
      
      # Add any other input widgets you need here
      
    ),
    mainPanel(
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
