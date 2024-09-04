import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import requests



credits=pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')

movies=movies.merge(credits, on='title',suffixes=('_movies', '_credits'))
movies=movies[['genres','id','keywords','title','cast','overview','crew']]


movies.dropna(inplace=True)


# Convert the string in 'genres' to an actual list of dictionaries
t = ast.literal_eval(movies.iloc[0].genres)

# Now, extract the 'name' from each dictionary
temp = [i['name'] for i in t]


def tolist(obj):
    temp = [i['name'] for i in ast.literal_eval(obj)]
    return temp

def tolist3(obj):
    temp = [i['character'] for i in ast.literal_eval(obj)]
    return temp

def tolist4(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l
            


movies['genres']=movies['genres'].apply(tolist)
movies['keywords']=movies['keywords'].apply(tolist)
movies['cast']=movies['cast'].apply(tolist3)
movies['crew']=movies['crew'].apply(tolist4)

movies['overview']=movies['overview'].apply(lambda x: x.split(' '))



movies['genres']=movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])


movies['tags']=movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']+movies['overview']




new_df=movies[['id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))



cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()



# Create an instance of the PorterStemmer
ps = PorterStemmer()

def stem(text):
    # Split the text into words, stem each word, and then join them back into a string
    stemmed_words = [ps.stem(word) for word in text.split()]
    return ' '.join(stemmed_words)


new_df['tags']=new_df['tags'].apply(stem)





similarity=cosine_similarity(vectors)


def recommend(movie):
    # Find the index of the movie in the DataFrame
    index = new_df[new_df['title'] == movie].index[0]  
    # Calculate similarity distances for the movie
    distances = similarity[index]
    
    # Sort movies based on similarity scores in descending order
    movies_list = sorted(
        list(enumerate(distances)),  # Creates a list of tuples (index, similarity score)
        reverse=True,                # Sorts the list in descending order
        key=lambda x: x[1]           # Sorts based on the similarity score (second element of the tuple)
    )
    
    # Recommend the top 5 most similar movies (ignoring the first one since it's the movie itself)
    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list[1:6]]
    for i in movies_list[1:6]:  # Start from index 1 to skip the movie itself
        print(new_df.iloc[i[0]].title)  # Access the title using the index in new_df
    return recommended_movies


def get_movies_list():
    return list(new_df['title'])



def get_poster(movie_title):
    URL=f'https://www.omdbapi.com/?apikey=e9409fce&t='
    response = requests.get(URL, params={'t': movie_title})
    data = response.json()
    return data.get('Poster')

