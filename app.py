import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

Movies = pd.read_csv('dataset/movies.csv') 
Credits = pd.read_csv('dataset/credits.csv') 

Credits.columns = ["id",'title', 'cast', 'crew']
Movies= Movies.merge(Credits, on="id")  

Movies=Movies.drop('title_x', axis=1) 
Movies.columns = ['budget', 'genres', 'homepage', 'id', 'keywords', 'original_language',
       'original_title', 'overview', 'popularity', 'production_companies',
       'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'vote_average', 'vote_count',
       'title', 'cast', 'crew'] 

#Credits, Genres and Keywords Based Recommender 

#parse the stringified features into their corresponding python objects 
from ast import literal_eval #safely evaluate an expression node or a string containing a Python literal
features = ['cast', 'crew', 'keywords', 'genres'] 
for feature in features:
    Movies[feature] = Movies[feature].apply(literal_eval)  

#get directors  
def getDirectors(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name'] 
    return np.nan 

#returns the top 3 of the list 
def getList(x):
    if isinstance(x, list):
        names=[i['name']for i in x]
        if len(names) > 3:
            return names[:3]
        return names
    return []

Movies['director'] = Movies['crew'].apply(getDirectors) 

f=['cast', 'keywords','genres'] 
for feature in f:
    Movies[feature] = Movies[feature].apply(getList) 

#clean data remove spaces and lowercase  

def cleanData(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:                                                   
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return '' 
features=['cast', 'keywords', 'director', 'genres']

for f in features:
    Movies[f]=Movies[f].apply(cleanData) 

def createSoup(x):
    return ''.join(x['keywords'])+' '+ ' '.join(x['cast']) + ' '+' '.join(x['director']) +' '+ ' '.join(x['genres']) 
Movies['soup'] = Movies.apply(createSoup, axis=1)

#countvectorizer 
from sklearn.feature_extraction.text import CountVectorizer  
count = CountVectorizer(stop_words='english') 
countMatrix = count.fit_transform(Movies['soup']) 

#cosine similarity matrix 
from sklearn.metrics.pairwise import cosine_similarity 
cosineSim2 = cosine_similarity(countMatrix, countMatrix)

#reset the index 
Movies = Movies.reset_index() 
indices = pd.Series(Movies.index, index=Movies['title'])  
       
def getRecomendation(title, consineSim=cosineSim2):
    idx=indices[title] #get the index of the movie 
    simScores = list(enumerate(consineSim[idx])) # get the pairwise similarity of all movies 
    simScores = sorted(simScores, key=lambda x:x[1], reverse=True) # sort the movies based on the similarity score 
    simScores = simScores[1:11]  # first 10 most similar movies 
    movieIndices = [i[0] for i in simScores] #movie indices 
    return Movies['title'].iloc[movieIndices] 

print(getRecomendation('The Godfather'))


