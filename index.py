import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as w
w.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

books.dropna(inplace=True)

books['Year-Of-Publication'].value_counts().index.values
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
books['Publication_Year'] = pd.to_datetime(books['Year-Of-Publication'], format='%Y', errors='coerce')
books.drop(columns=['Year-Of-Publication'], inplace=True)
books = pd.DataFrame(books)

books['Year-Of-Publication'] = books['Publication_Year'].dt.year

#Remove all the invalid years
books = books[~(books['Year-Of-Publication'] == 2037)]
books = books[~(books['Year-Of-Publication'] == 2026)]
books = books[~(books['Year-Of-Publication'] == 2030)]
books = books[~(books['Year-Of-Publication'] == 2050)]
books = books[~(books['Year-Of-Publication'] == 2038)]

users.drop('Age',axis=1,inplace=True)

users['Location'] = users['Location'].apply(lambda x:x.split(',')[-1])

#Not all the users have rated fo all the books we choose a threshold, users only who have rated more than 100 times

x = ratings['User-ID'].value_counts() > 200
y = x[x].index

ratings = ratings[ratings['User-ID'].isin(y)]
rating_with_books = ratings.merge(books, on='ISBN')

number_rating = rating_with_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
number_rating.rename(columns= {'Book-Rating':'number_of_ratings'}, inplace=True)

final_rating = rating_with_books.merge(number_rating, on='Book-Title')
final_rating = final_rating[final_rating['number_of_ratings'] >= 25]
final_rating.drop_duplicates(['User-ID','Book-Title'], inplace=True)

#create a pivot table
book_pivot = final_rating.pivot_table(columns='User-ID', index='Book-Title', values="Book-Rating")
book_pivot.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(book_pivot)


def recommend(book_name):
    # index fetch
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

recommend('Harry Potter and the Chamber of Secrets (Book 2)')