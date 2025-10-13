import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Open Data Culinary Road Trip",
    page_icon="🍽️",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load the European restaurants dataset"""
    df = pd.read_csv('european_restaurants.csv')
    return df

# Main title
st.title("🍽️ Open Data Culinary Road Trip")
st.markdown("*Discover trending restaurants, find the best spots nearby, and plan a foodie road trip across Europe!*")
st.write("Bienvenue dans votre exploration culinaire en Europe à partir de données **Open Data** 🍷🇫🇷🇮🇹🇪🇸")

# Load the data
df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Trending Restaurants", "Nearby Restaurants Map", "Road Trip Planner"])

# ===========================
# PAGE 1: TRENDING RESTAURANTS
# ===========================
if page == "Trending Restaurants":
    st.header("📈 Trending Restaurants")
    st.markdown("Discover the most popular and highly-rated restaurants across Europe")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        country_filter = st.multiselect(
            "Filter by Country",
            options=sorted(df['country'].unique()),
            default=None
        )
    with col2:
        cuisine_filter = st.multiselect(
            "Filter by Cuisine",
            options=sorted(df['cuisine'].unique()),
            default=None
        )
    
    # Apply filters
    filtered_df = df.copy()
    if country_filter:
        filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]
    if cuisine_filter:
        filtered_df = filtered_df[filtered_df['cuisine'].isin(cuisine_filter)]
    
    # Sort by rating and reviews
    trending_df = filtered_df.sort_values(['rating', 'reviews_count'], ascending=[False, False]).head(20)
    
    # Display metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Restaurants", len(filtered_df))
    with c2:
        st.metric("Average Rating", f"{filtered_df['rating'].mean():.2f}")
    with c3:
        st.metric("Countries", filtered_df['country'].nunique())
    
    # Chart: Top countries by restaurant count
    st.subheader("Top Countries by Restaurant Count")
    country_counts = filtered_df['country'].value_counts().head(10)
    fig = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        labels={'x': 'Number of Restaurants', 'y': 'Country'},
        color=country_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Chart: Cuisine distribution
    st.subheader("Cuisine Type Distribution")
    cuisine_counts = filtered_df['cuisine'].value_counts()
    fig2 = px.pie(
        values=cuisine_counts.values,
        names=cuisine_counts.index,
        title="",
        hole=0.4
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Top trending restaurants table (cards)
    st.subheader("🌟 Top Trending Restaurants")
    for _, row in trending_df.iterrows():
        with st.container():
            cc1, cc2, cc3 = st.columns([3, 1, 1])
            with cc1:
                st.markdown(f"### {row['name']}")
                st.markdown(f"📍 {row['city']}, {row['country']} | 🍴 {row['cuisine']}")
                st.markdown(f"📞 {row['phone']} | 📧 {row['address']}")
            with cc2:
                st.metric("Rating", f"⭐ {row['rating']}")
                st.markdown(f"💰 {row['price_range']}")
            with cc3:
                st.metric("Reviews", f"{row['reviews_count']:,}")
            st.divider()

# ===========================
# PAGE 2: NEARBY RESTAURANTS MAP
# ===========================
elif page == "Nearby Restaurants Map":
    st.header("🗺️ Nearby Restaurants Map")
    st.markdown("Explore restaurants on an interactive map with advanced filters")
    
    # Filters in sidebar
    st.sidebar.subheader("Map Filters")
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=sorted(df['country'].unique()),
        default=None
    )
    selected_cuisines = st.sidebar.multiselect(
        "Select Cuisines",
        options=sorted(df['cuisine'].unique()),
        default=None
    )
    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=float(df['rating'].min()),
        max_value=float(df['rating'].max()),
        value=float(df['rating'].min()),
        step=0.1
    )
    price_ranges = st.sidebar.multiselect(
        "Price Range",
        options=sorted(df['price_range'].unique()),
        default=None
    )
    
    # Apply filters
    map_df = df.copy()
    if selected_countries:
        map_df = map_df[map_df['country'].isin(selected_countries)]
    if selected_cuisines:
        map_df = map_df[map_df['cuisine'].isin(selected_cuisines)]
    map_df = map_df[map_df['rating'] >= min_rating]
    if price_ranges:
        map_df = map_df[map_df['price_range'].isin(price_ranges)]
    
    st.info(f"Showing {len(map_df)} restaurants based on your filters")
    
    if len(map_df) > 0:
        center_lat = map_df['latitude'].mean()
        center_lon = map_df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='OpenStreetMap')
        
        # Add markers
        for _, row in map_df.iterrows():
            if row['rating'] >= 4.7:
                color = 'red'
            elif row['rating'] >= 4.5:
                color = 'orange'
            elif row['rating'] >= 4.0:
                color = 'green'
            else:
                color = 'blue'
            popup_html = f"""
            <div style="width: 200px;">
                <h4>{row['name']}</h4>
                <p><b>City:</b> {row['city']}, {row['country']}</p>
                <p><b>Cuisine:</b> {row['cuisine']}</p>
                <p><b>Rating:</b> ⭐ {row['rating']}</p>
                <p><b>Price:</b> {row['price_range']}</p>
                <p><b>Reviews:</b> {row['reviews_count']:,}</p>
                <p><b>Phone:</b> {row['phone']}</p>
            </div>
            """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row['name'],
                icon=folium.Icon(color=color, icon='cutlery', prefix='fa')
            ).add_to(m)
        
        st_folium(m, width=1400, height=600)
        st.subheader("Restaurant Details")
        display_df = map_df[['name', 'country', 'city', 'cuisine', 'rating', 'price_range', 'reviews_count']].copy()
        display_df = display_df.sort_values('rating', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No restaurants match your filter criteria. Please adjust your filters.")

# ===========================
# PAGE 3: ROAD TRIP PLANNER
# ===========================
elif page == "Road Trip Planner":
    st.header("🚗 Multi-Day Foodie Road Trip Planner")
    st.markdown("Plan your perfect culinary journey across Europe!")
    
    # Trip parameters
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trip Settings")
        num_days = st.number_input("Number of Days", min_value=1, max_value=30, value=5, step=1)
        meals_per_day = st.number_input("Meals per Day", min_value=1, max_value=3, value=2, step=1)
        daily_budget = st.number_input("Daily Budget (€)", min_value=10, max_value=1000, value=100, step=10)
    with c2:
        st.subheader("Preferences")
        preferred_countries = st.multiselect(
            "Countries to Visit",
            options=sorted(df['country'].unique()),
            default=['France', 'Italy', 'Spain']
        )
        preferred_cuisines = st.multiselect(
            "Preferred Cuisines",
            options=sorted(df['cuisine'].unique()),
            default=None
        )
        min_rating_trip = st.slider("Minimum Restaurant Rating", min_value=3.0, max_value=5.0, value=4.0, step=0.1)
    
    if st.button("🎯 Generate Road Trip", type="primary"):
        trip_df = df.copy()
        if preferred_countries:
            trip_df = trip_df[trip_df['country'].isin(preferred_countries)]
        if preferred_cuisines:
            trip_df = trip_df[trip_df['cuisine'].isin(preferred_cuisines)]
        trip_df = trip_df[trip_df['rating'] >= min_rating_trip]
        
        price_to_cost = {'$': 20, '$$': 40, '$$$': 80, '$$$$': 150}
        trip_df['estimated_cost'] = trip_df['price_range'].map(price_to_cost)
        trip_df = trip_df[trip_df['estimated_cost'] <= (daily_budget / meals_per_day)]
        
        needed = num_days * meals_per_day
        if len(trip_df) < needed:
            st.warning(f"⚠️ Not enough restaurants match your criteria. Found {len(trip_df)} restaurants but need {needed}. Try adjusting your filters.")
        else:
            trip_df = trip_df.sort_values(['rating', 'reviews_count'], ascending=[False, False])
            selected_restaurants = []
            cities_visited = set()
            for _ in range(needed):
                available = trip_df[~trip_df['city'].isin(cities_visited)]
                if len(available) == 0:
                    available = trip_df
                r = available.iloc[0]
                selected_restaurants.append(r)
                cities_visited.add(r['city'])
                trip_df = trip_df[trip_df['name'] != r['name']]
                if len(trip_df) == 0:
                    break
            
            st.success(f"✅ Generated a {num_days}-day road trip with {len(selected_restaurants)} meals!")
            total_cost = sum([r['estimated_cost'] for r in selected_restaurants])
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                st.metric("Total Meals", len(selected_restaurants))
            with cc2:
                st.metric("Estimated Cost", f"€{total_cost}")
            with cc3:
                countries_visited = len(set([r['country'] for r in selected_restaurants]))
                st.metric("Countries", countries_visited)
            with cc4:
                avg_rating = np.mean([r['rating'] for r in selected_restaurants])
                st.metric("Avg Rating", f"⭐ {avg_rating:.2f}")
            
            st.subheader("📅 Your Itinerary")
            for day in range(num_days):
                st.markdown(f"### Day {day + 1}")
                start_i = day * meals_per_day
                end_i = min((day + 1) * meals_per_day, len(selected_restaurants))
                for i in range(start_i, end_i):
                    if i < len(selected_restaurants):
                        r = selected_restaurants[i]
                        meal_num = (i % meals_per_day) + 1
                        with st.container():
                            k1, k2, k3 = st.columns([3, 1, 1])
                            with k1:
                                st.markdown(f"**Meal {meal_num}: {r['name']}**")
                                st.markdown(f"📍 {r['city']}, {r['country']} | 🍴 {r['cuisine']}")
                                st.markdown(f"📧 {r['address']}")
                            with k2:
                                st.markdown(f"⭐ **{r['rating']}**")
                                st.markdown(f"💰 {r['price_range']} (~€{r['estimated_cost']})")
                            with k3:
                                st.markdown(f"📞 {r['phone']}")
                st.divider()
            
            st.subheader("🗺️ Trip Map")
            trip_restaurants_df = pd.DataFrame(selected_restaurants)
            center_lat = trip_restaurants_df['latitude'].mean()
            center_lon = trip_restaurants_df['longitude'].mean()
            trip_map = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='OpenStreetMap')
            
            for idx, r in enumerate(selected_restaurants):
                folium.Marker(
                    location=[r['latitude'], r['longitude']],
                    popup=f"<b>Stop {idx+1}: {r['name']}</b><br>{r['city']}, {r['country']}",
                    tooltip=f"Stop {idx+1}: {r['name']}",
                    icon=folium.DivIcon(html=f"""
                        <div style="background-color: #FF4B4B; 
                                    color: white; 
                                    border-radius: 50%; 
                                    width: 30px; 
                                    height: 30px; 
                                    display: flex; 
                                    align-items: center; 
                                    justify-content: center;
                                    font-weight: bold;
                                    font-size: 14px;
                                    border: 2px solid white;">
                            {idx+1}
                        </div>
                    """)
                ).add_to(trip_map)
            
            coordinates = [[r['latitude'], r['longitude']] for r in selected_restaurants]
            folium.PolyLine(coordinates, color='#FF4B4B', weight=3, opacity=0.7, popup='Trip Route').add_to(trip_map)
            st_folium(trip_map, width=1400, height=500)

# Footer
st.markdown("---")
st.caption("Projet Open Data — MIASHS 2025")
st.markdown("*Data source: Tripadvisor European Restaurants Open Data*")