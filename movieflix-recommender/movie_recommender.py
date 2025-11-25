import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
import altair as alt
import requests
import re 

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Movieflix AI", layout="wide", page_icon="🍿")

# ==========================================
# 🔑 API KEY CONFIGURATION
# ==========================================
TMDB_API_KEY = "TMDB_API_KEY" 

# --- 1. DATA LOADING & API FUNCTIONS ---
@st.cache_data
def fetch_movie_data(tmdbId):
    """Fetches movie poster and overview from TMDB API."""
    default_img = "https://via.placeholder.com/500x750?text=No+Image"
    default_overview = "No description available."
    
    if pd.isna(tmdbId): return default_img, default_overview
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdbId)}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=3)
        if response.status_code != 200: return default_img, default_overview
        
        data = response.json()
        poster_path = data.get('poster_path')
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else default_img
        overview = data.get('overview', default_overview)
        return full_path, overview
    except Exception as e: 
        return default_img, default_overview

@st.cache_resource
def load_data():
    """Loads and preprocesses the MovieLens dataset."""
    path = 'ml-latest-small/' 
    try:
        movies = pd.read_csv(path + 'movies.csv')
        ratings = pd.read_csv(path + 'ratings.csv')
        tags = pd.read_csv(path + 'tags.csv')
        links = pd.read_csv(path + 'links.csv')
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure 'ml-latest-small' folder exists.")
        return None, None
    
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies = pd.merge(movies, movie_tags, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna('')
    movies['genres_str'] = movies['genres'].str.replace('|', ' ')
    movies['metadata_soup'] = movies['genres_str'] + ' ' + movies['tag']
    movies = pd.merge(movies, links, on='movieId', how='left')
    
    vote_counts = ratings.groupby('movieId').count()['rating'].reset_index()
    vote_counts.columns = ['movieId', 'num_votes']
    movies = pd.merge(movies, vote_counts, on='movieId', how='left')
    
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
    
    movies['num_votes'] = movies['num_votes'].fillna(0)
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    
    return movies, ratings

movies, ratings = load_data()

# --- 2. MODEL TRAINING ---
@st.cache_resource
def train_models(movies, ratings):
    if movies is None: return None, None, None

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['metadata_soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    
    return cosine_sim, indices, svd

cosine_sim, indices, svd = train_models(movies, ratings)

# --- 3. SESSION STATE & CALLBACKS (AUTOMATION LOGIC) ---
# Initialize session state for filters if not present
if 'min_rating_val' not in st.session_state:
    st.session_state.min_rating_val = 2.5 
if 'min_votes_val' not in st.session_state:
    st.session_state.min_votes_val = 20   

def update_filters():
    """Updates the sliders based on the selected Sort Option"""
    option = st.session_state.sort_radio
    
    if "Similarity" in option:
        st.session_state.min_rating_val = 0.0
        st.session_state.min_votes_val = 0
    elif "Quality" in option:
        st.session_state.min_rating_val = 3.5
        st.session_state.min_votes_val = 50
    else: 
        st.session_state.min_rating_val = 2.5
        st.session_state.min_votes_val = 20

# --- 4. SIDEBAR UI ---
if movies is not None:
    with st.sidebar:
        st.header("🎛️ Control Center")
        user_id = st.number_input("User ID", 1, 610, 1)
        selected_movie = st.selectbox("Select a movie you like:", movies['title'].values)
        
        st.divider()
        st.subheader("⚙️ Recommendation Logic")
        
        # Radio Button with Callback
        sort_option = st.radio(
            "Sort by:",
            ("✨ Balanced (Smart Mix)", "🧩 Similarity (Thematic Match)", "⭐ Quality (Predicted Rating)"),
            index=0,
            key="sort_radio",
            on_change=update_filters, 
            help="Balanced combines both. Similarity finds sequels. Quality finds top-rated movies."
        )
        
        st.divider()
        st.subheader("🎯 Filters (Auto-Adjusting)")
        
        # Sliders connected to Session State
        min_rating = st.slider(
            "Min Predicted Rating ⭐", 0.0, 5.0, 
            value=st.session_state.min_rating_val, 
            step=0.1,
            key="min_rating_slider"
        )
        
        min_votes = st.slider(
            "Min Votes 👥", 0, 500, 
            value=st.session_state.min_votes_val, 
            step=10,
            key="min_votes_slider"
        )
        
        # Update session state if user moves slider manually
        st.session_state.min_rating_val = min_rating
        st.session_state.min_votes_val = min_votes
        
        unique_genres = set('|'.join(movies['genres']).split('|'))
        genre_filter = st.selectbox("Genre Filter 🎭", ["All"] + list(sorted(unique_genres)))

# --- 5. RECOMMENDATION ENGINE ---
def get_recommendations(user_id, movie_title):
    if movie_title not in indices: return None
    idx = indices[movie_title]
    
    source_year = movies.iloc[idx]['year']
    source_genres = movies.iloc[idx]['genres']
    is_source_animation = 'Animation' in source_genres
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Keyword Matching Setup
    clean_input = re.sub(r'\(\d{4}\)', '', movie_title.lower())
    input_words = set(re.findall(r'\b\w+\b', clean_input))
    stop_words = {'the', 'a', 'an', 'of', 'and', 'in', 'for', 'to', 'at', 'on', 'with', 'by', 'part', 'ii', 'iii'}
    search_keywords = input_words - stop_words
    
    boosted_scores = []
    
    for i, score in sim_scores:
        if i == idx: continue
        
        bonus = 0
        target_year = movies.iloc[i]['year']
        
        # A. TITLE BOOSTING
        target_title = movies.iloc[i]['title'].lower()
        clean_target = re.sub(r'\(\d{4}\)', '', target_title)
        target_words = set(re.findall(r'\b\w+\b', clean_target))
        common_words = search_keywords.intersection(target_words)
        
        if common_words:
            bonus += len(common_words) * 5.0
            if 'avengers' in common_words: bonus += 15.0
            if 'harry' in common_words and 'potter' in common_words: bonus += 10.0
            if 'star' in common_words and 'wars' in common_words: bonus += 10.0
            
        # B. YEAR & ERA LOGIC
        if pd.notnull(source_year) and pd.notnull(target_year):
            diff = abs(source_year - target_year)
            if diff <= 3: bonus += 3.0
            elif diff <= 6: bonus += 1.5
            if source_year >= 2008 and target_year < 2005: bonus -= 20.0
                
        # C. GENRE PROTECTION
        target_genres = movies.iloc[i]['genres']
        is_target_animation = 'Animation' in target_genres
        if not is_source_animation and is_target_animation: bonus -= 10.0 
            
        boosted_scores.append((i, score + bonus))
    
    # Sorting
    boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)
    boosted_scores = boosted_scores[:150]
    
    movie_indices = [i[0] for i in boosted_scores]
    similarity_values = [i[1] for i in boosted_scores]
    
    recs = movies.iloc[movie_indices].copy()
    recs['similarity_score'] = similarity_values
    recs['est_rating'] = recs['movieId'].apply(lambda x: svd.predict(user_id, x).est)
    recs['hybrid_score'] = recs['similarity_score'] + recs['est_rating']
    
    # Apply Filters
    if genre_filter != "All": recs = recs[recs['genres'].str.contains(genre_filter)]
    recs = recs[recs['num_votes'] >= min_votes]
    recs = recs[recs['est_rating'] >= min_rating]
    
    # FINAL SORTING LOGIC
    if "Quality" in sort_option:
        recs = recs.sort_values('est_rating', ascending=False)
    elif "Similarity" in sort_option:
        recs = recs.sort_values('similarity_score', ascending=False)
    else: 
        recs = recs.sort_values('hybrid_score', ascending=False)
        
    return recs.head(10)

# --- 6. MAIN UI LAYOUT ---
st.title("🍿 Movieflix AI")
st.markdown("### Intelligent Hybrid Recommendation System")

if movies is not None:
    if st.button("✨ Get Recommendations ✨", type="primary"):
        with st.spinner('Crunching numbers...'):
            recs = get_recommendations(user_id, selected_movie)
        
        if recs is not None and not recs.empty:
            st.success(f"Showing top matches for **{selected_movie}**")
            
            tab1, tab2 = st.tabs(["🎬 Movies", "📊 Analytics"])
            
            with tab1:
                for i, row in recs.iterrows():
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        poster_url, overview_text = fetch_movie_data(row['tmdbId'])
                        
                        with col1:
                            st.image(poster_url, use_container_width=True)
                        with col2:
                            title_display = f"{i+1}. {row['title']}"
                            if pd.notnull(row['year']): title_display += f" ({int(row['year'])})"
                            st.subheader(title_display)
                            st.caption(f"**Genres:** {row['genres'].replace('|', ', ')}")
                            
                            # Debug info depending on mode
                            if "Similarity" in sort_option:
                                 st.markdown(f"🎯 **Match Score:** `{row['similarity_score']:.2f}`")
                            elif "Balanced" in sort_option:
                                 st.markdown(f"⚖️ **Hybrid Score:** `{row['hybrid_score']:.2f}`")

                            with st.expander("📖 Plot Overview", expanded=True):
                                st.write(overview_text)
                            
                            c1, c2 = st.columns(2)
                            with c1: st.metric("Your Prediction", f"⭐ {row['est_rating']:.1f}/5")
                            with c2: st.metric("Audience Score", f"👥 {row['avg_rating']:.1f} ({int(row['num_votes'])})")
                            
                            if pd.notnull(row['imdbId']):
                                st.markdown(f"[👉 View on IMDB](https://www.imdb.com/title/tt{int(row['imdbId']):07d}/)")
                        st.divider()

            with tab2:
                st.subheader("Recommendation Insights")
                all_genres = []
                for g in recs['genres']: all_genres.extend(g.split('|'))
                genre_df = pd.DataFrame(all_genres, columns=['Genre']).value_counts().reset_index()
                genre_df.columns = ['Genre', 'Count']
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Genre Distribution**")
                    st.altair_chart(alt.Chart(genre_df).mark_arc(innerRadius=50).encode(
                       theta=alt.Theta("Count", stack=True), 
                       color=alt.Color("Genre"), 
                       tooltip=["Genre", "Count"]
                    ), use_container_width=True)
                
                with c2:
                    st.markdown("**Prediction vs Global Average**")
                    chart_data = recs[['title', 'est_rating', 'avg_rating']].melt('title', var_name='Metric', value_name='Rating')
                    st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
                       x=alt.X('Rating', scale=alt.Scale(domain=[0, 5])), 
                       y=alt.Y('title', sort='-x'), 
                       color='Metric', 
                       tooltip=['title', 'Rating']
                    ).interactive(), use_container_width=True)
        else:
            st.error("No movies found! The automatic filters might be too strict for this specific movie.")