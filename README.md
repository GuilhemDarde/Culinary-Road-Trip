# Culinary-Road-Trip
Interactive Streamlit app powered by Tripadvisor and open data. Discover trending restaurants, find the best spots nearby, and plan a foodie road trip across Europe tailored to your tastes, budget, and travel days.

## Features

🍽️ **Trending Restaurants**: Discover the most popular and highly-rated restaurants across Europe with interactive visualizations and filters.

🗺️ **Nearby Restaurants Map**: Explore restaurants on an interactive map with advanced filtering by country, cuisine, rating, and price range.

🚗 **Road Trip Planner**: Plan your perfect multi-day culinary journey across Europe with customizable budget, cuisine preferences, and daily itinerary.

## Link
app : https://guilhemdarde-culinary-road-trip-accueil-qbjf1j.streamlit.app
HuggingFace dataset : https://huggingface.co/datasets/Amoham16/dataset-resto-10k

## Installation

1. Clone this repository:
```bash
git clone https://github.com/GuilhemDarde/Culinary-Road-Trip.git
cd Culinary-Road-Trip
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Dataset

The app uses a curated dataset of 49 European restaurants from Tripadvisor, spanning 9 countries:
- France, Italy, Spain, Germany, United Kingdom
- Denmark, Belgium, Austria, Portugal

Each restaurant includes:
- Name, location (city, country, coordinates)
- Cuisine type, rating, price range
- Number of reviews, contact information

## Technologies

- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **Folium**: Interactive maps
- **Plotly**: Data visualizations
- **NumPy**: Numerical computations
