import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Page configuration
st.set_page_config(page_title="Book Recommender System", page_icon="📚", layout="wide")

# Cache data loading and model training to optimize performance
@st.cache_resource
def load_and_train_model():
    # 1. Load Datasets
    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin1', on_bad_lines='skip')
    books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)

    # 2. Preprocess & Filter Sparsity
    # Keep explicit ratings only (scale 1-10)
    explicit_ratings = ratings[ratings['Book-Rating'] > 0]

    # Filter users who have rated at least 50 books
    user_counts = explicit_ratings['User-ID'].value_counts()
    active_users = user_counts[user_counts >= 50].index

    # Filter books that have received at least 30 ratings
    book_counts = explicit_ratings['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= 30].index

    # Apply filter
    filtered_df = explicit_ratings[
        explicit_ratings['User-ID'].isin(active_users) & 
        explicit_ratings['ISBN'].isin(popular_books)
    ]

    # 3. Create User-Item Matrix & Convert to Sparse Matrix
    user_book_matrix = filtered_df.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
    sparse_matrix = csr_matrix(user_book_matrix.values)

    # 4. Train Item-Based k-NN Model (Cosine Similarity)
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(sparse_matrix)

    # Map valid ISBNs to Book Titles for the dropdown UI
    valid_books = books[books['ISBN'].isin(user_book_matrix.index)][['ISBN', 'Book-Title', 'Book-Author']].drop_duplicates('ISBN')
    
    return user_book_matrix, knn_model, books, valid_books

# Recommendation Inference Function
def recommend_books(isbn, user_book_matrix, knn_model, books, num_recommendations=5):
    query_idx = user_book_matrix.index.get_loc(isbn)
    distances, indices = knn_model.kneighbors(
        user_book_matrix.iloc[query_idx, :].values.reshape(1, -1), 
        n_neighbors=num_recommendations + 1
    )
    
    rec_isbns = [user_book_matrix.index[i] for i in indices.flatten()[1:]]
    rec_distances = distances.flatten()[1:]
    
    # Format output DataFrame
    recommendations_df = pd.DataFrame({
        'ISBN': rec_isbns, 
        'Cosine Distance': rec_distances
    })
    
    result = recommendations_df.merge(
        books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']], 
        on='ISBN', 
        how='left'
    ).drop_duplicates('ISBN')
    
    return result

# --- STREAMLIT UI ---
st.title("📚 Item-Based Book Recommender")
st.markdown("Find similar books using collaborative filtering ($k$-Nearest Neighbors with Cosine Similarity).")

with st.spinner("Loading datasets and training k-NN model..."):
    user_book_matrix, knn_model, books, valid_books = load_and_train_model()

# Sidebar Controls
st.sidebar.header("Recommendation Settings")
num_recs = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=15, value=5)

# Book Selection Dropdown
valid_books['Display_Label'] = valid_books['Book-Title'].fillna('Unknown Title') + " — " + valid_books['ISBN']
selected_label = st.selectbox(
    "Search or select a book you enjoyed:",
    options=valid_books['Display_Label'].tolist()
)

# Extract selected ISBN
selected_isbn = valid_books[valid_books['Display_Label'] == selected_label]['ISBN'].values[0]
selected_book_info = valid_books[valid_books['ISBN'] == selected_isbn].iloc[0]

# Trigger Recommendations
if st.button("Get Recommendations", type="primary"):
    st.subheader(f"Target Book: {selected_book_info['Book-Title']}")
    st.caption(f"**Author:** {selected_book_info['Book-Author']} | **ISBN:** `{selected_isbn}`")
    
    st.divider()
    st.subheader(f"Top {num_recs} Recommended Books")
    
    recs = recommend_books(selected_isbn, user_book_matrix, knn_model, books, num_recommendations=num_recs)
    
    # Display results cleanly in an interactive Streamlit table
    st.dataframe(
        recs[['Book-Title', 'Book-Author', 'ISBN', 'Cosine Distance']],
        use_container_width=True,
        hide_index=True
    )
