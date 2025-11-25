import streamlit as st
import pickle
import pandas as pd
import difflib
from sklearn.metrics.pairwise import linear_kernel # Kita butuh ini di App sekarang

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
# Load daftar film
movie_dict = pickle.load(open('movie_list.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

# Load Matrix TF-IDF (Bukan similarity matrix yg 1.9GB)
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))

# ---------------------------------------------------------
# 2. FUNGSI REKOMENDASI (Hitung on-the-fly)
# ---------------------------------------------------------
def recommend(movie_title):
    # A. Cari Judul (Fuzzy Match)
    list_of_all_titles = movies['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_title, list_of_all_titles, n=1)
    
    if not find_close_match:
        return ["Film tidak ditemukan."]
    
    closest_title = find_close_match[0]
    
    # B. Ambil Index Film
    idx = movies[movies['title'] == closest_title].index[0]
    
    # C. HITUNG KEMIRIPAN SAAT ITU JUGA (Hemat Memori)
    # Kita hanya menghitung jarak antara Film Input (idx) vs Semua Film
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
    
    # D. Ambil Skor
    # linear_kernel mengembalikan array [[skor1, skor2, ...]]
    sim_scores = list(enumerate(cosine_sim[0]))
    
    # E. Urutkan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # F. Ambil 5 Teratas (Skip index 0 karena itu diri sendiri)
    recommended_movies = []
    for i in range(1, 6):
        res_idx = sim_scores[i][0]
        recommended_movies.append(movies.iloc[res_idx].title)
        
    return recommended_movies

# ---------------------------------------------------------
# 3. TAMPILAN WEB
# ---------------------------------------------------------
st.title('🎬 Rekomendasi Film')

selected_movie_name = st.selectbox(
    'Pilih Film Favoritmu:',
    movies['title'].values
)

if st.button('Cari Rekomendasi'):
    with st.spinner('Menghitung kemiripan...'):
        hasil = recommend(selected_movie_name)
        for i, film in enumerate(hasil):
            st.write(f"{i+1}. {film}")
