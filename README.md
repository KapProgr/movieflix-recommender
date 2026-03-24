Movieflix AI: Intelligent Hybrid Recommender System
An advanced movie recommendation engine built with Python and Streamlit. It combines Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF) to provide personalized movie suggestions.
This project goes beyond basic recommendations by implementing a custom "Smart Boosting" algorithm that detects franchises (e.g., Avengers, Harry Potter) and handles time-decay logic.
Live Demo
Click here to view the App
Key Features

Hybrid Engine: Combines user voting patterns (SVD Matrix Factorization) with metadata similarity (Cosine Similarity).
Balanced Logic: A "Smart Mix" mode that automatically balances between movie quality and thematic similarity.
Smart Title Boosting: Custom Regex logic to detect franchises (e.g., Marvel, Star Wars) and prioritize sequels.
Time-Aware: Implements "Era Penalty" to distinguish between classic and modern reboots (e.g., filtering out 1998 Avengers when looking for MCU).
Genre Protection: "Anti-Cartoon" logic to separate Live Action from Animation matches.
Analytics Dashboard: Interactive charts showing genre distribution and prediction vs. global average.
TMDB API Integration: Fetches real-time posters and plot overviews.

Tech Stack

Python 3.10+
Streamlit (UI/UX)
Scikit-Learn (TF-IDF, Cosine Similarity)
Scikit-Surprise (SVD Algorithm)
Pandas & NumPy (Data Manipulation)
Altair (Data Visualization)
TMDB API (Movie Metadata)

Installation

Clone the repository:

bashgit clone https://github.com/KapProgr/movieflix-recommender.git
cd movieflix-recommender

Install dependencies:

bashpip install -r requirements.txt

Set up API Key:

Get a free API Key from TMDB.
Create a .streamlit/secrets.toml file (for local dev) or set it up in Streamlit Cloud secrets:



tomlTMDB_API_KEY = "your_api_key_here"

Run the app:

bashpython -m streamlit run movie_recommender.py
Dataset
This project uses the MovieLens Latest Small dataset.

movies.csv
ratings.csv
tags.csv
links.csv

Contributing
Feel free to fork this repository and submit pull requests.
License
This project is for educational purposes.
Feel free to fork this repository and submit pull requests.

## 📜 License
This project is for educational purposes.
