import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Check if environment variables are loaded
if not CLIENT_ID or not CLIENT_SECRET:
    st.error("⚠️ Spotify API credentials not found! Please check your .env file.")
    st.stop()

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        artist = music.iloc[i[0]].artist
        print(artist)
        print(music.iloc[i[0]].song)
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
        recommended_music_names.append(music.iloc[i[0]].song)

    return recommended_music_names,recommended_music_posters

st.header('Music Recommender System')

# Add a subtitle with emojis
st.markdown("### 🎧 Discover Your Next Favorite Song! 🎶")
st.markdown("*Powered by machine learning and Spotify API*")

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header */
    h1 {
        color: #ffffff !important;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        padding: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.4);
    }
    
    .stButton > button:hover {
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(238, 90, 36, 0.6);
    }
    
    /* Column styling for recommendations */
    .stColumn {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 0.6rem;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Text styling */
    .stText {
        color: #ffffff !important;
        font-weight: bold;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Image styling */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .stImage:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)
music = pickle.load(open('dataframe','rb'))
similarity = pickle.load(open('similarity','rb'))

music_list = music['song'].values

st.markdown("### 🎵 Choose a song you like:")
selected_movie = st.selectbox(
    "Type or select a song from the dropdown",
    music_list,
    help="Select a song to get personalized recommendations based on your taste!"
)

if st.button('🎵 Show Recommendations 🎵'):
    recommended_music_names,recommended_music_posters = recommend(selected_movie)
    
    st.markdown("### 🎶 Recommended Songs for You:")
    st.markdown("---")
    
    # Create 5 columns for recommendations in one row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"<div class='stText'>{recommended_music_names[0]}</div>", unsafe_allow_html=True)
        st.image(recommended_music_posters[0], use_container_width=True)
        
    with col2:
        st.markdown(f"<div class='stText'>{recommended_music_names[1]}</div>", unsafe_allow_html=True)
        st.image(recommended_music_posters[1], use_container_width=True)
        
    with col3:
        st.markdown(f"<div class='stText'>{recommended_music_names[2]}</div>", unsafe_allow_html=True)
        st.image(recommended_music_posters[2], use_container_width=True)
        
    with col4:
        st.markdown(f"<div class='stText'>{recommended_music_names[3]}</div>", unsafe_allow_html=True)
        st.image(recommended_music_posters[3], use_container_width=True)
        
    with col5:
        st.markdown(f"<div class='stText'>{recommended_music_names[4]}</div>", unsafe_allow_html=True)
        st.image(recommended_music_posters[4], use_container_width=True)




