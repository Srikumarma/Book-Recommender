import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your datasets (books, ratings)
# Replace these lines with loading your actual datasets
# books = pd.read_csv('books.csv')
# ratings = pd.read_csv('ratings.csv')

# ... your existing code to preprocess the data ...
books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

books.dropna(inplace=True)

books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
books['Publication_Year'] = pd.to_datetime(books['Year-Of-Publication'], format='%Y', errors='coerce')
books.drop(columns=['Year-Of-Publication'], inplace=True)
books = pd.DataFrame(books)

books['Year-Of-Publication'] = books['Publication_Year'].dt.year

# Remove all the invalid years
books = books[~(books['Year-Of-Publication'] == 2037)]
books = books[~(books['Year-Of-Publication'] == 2026)]
books = books[~(books['Year-Of-Publication'] == 2030)]
books = books[~(books['Year-Of-Publication'] == 2050)]
books = books[~(books['Year-Of-Publication'] == 2038)]

users.drop('Age', axis=1, inplace=True)

users['Location'] = users['Location'].apply(lambda x: x.split(',')[-1])

# Create a Streamlit app
st.title('Book Recommendation System')

# Create a dropdown to select a book title
book_names = books['Book-Title'].tolist()
book_name = st.selectbox('Select a Book Title:', book_names)

# Debugging print statements
print("Unique Book Titles:", books['Book-Title'].unique())
print("Selected Book Name:", book_name)

x = ratings['User-ID'].value_counts() > 200
y = x[x].index

ratings = ratings[ratings['User-ID'].isin(y)]
rating_with_books = ratings.merge(books, on='ISBN')

number_rating = rating_with_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
number_rating.rename(columns={'Book-Rating': 'number_of_ratings'}, inplace=True)

final_rating = rating_with_books.merge(number_rating, on='Book-Title')
final_rating = final_rating[final_rating['number_of_ratings'] >= 25]
final_rating.drop_duplicates(['User-ID', 'Book-Title'], inplace=True)

# create a pivot table
book_pivot = final_rating.pivot_table(columns='User-ID', index='Book-Title', values="Book-Rating")
book_pivot.fillna(0, inplace=True)

# from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(book_pivot)

# Define a function to recommend books
import numpy as np

def recommend(book_name):
    if book_name in book_pivot.index:
        index = np.where(book_pivot.index == book_name)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == book_pivot.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)

        return data
    else:
        return []

# Usage
book_recommendations = recommend(book_name)
if book_recommendations:
    for rec in book_recommendations:
        print(f"Title: {rec[0]}")
        print(f"Author: {rec[1]}")
        print(f"Image URL: {rec[2]}")
else:
    print("Book not found in the dataset.")
    st.write("Book not found in the dataset.")

if st.button('Recommend'):
    recommendations = recommend(book_name)
    st.subheader('Recommended Books:')
    for rec in recommendations:
        st.write(f"Title: {rec[0]}")
        st.write(f"Author: {rec[1]}")
        st.image(rec[2], use_column_width=True)
