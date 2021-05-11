#!/usr/bin/env python3
# coding: utf-8

# Pipeline for this case
# 1. Initial Imports
# 2. Loading the Data
# 3. Processing the Data
#    - Processing User IDs
#    - Processing Movie IDs
#    - Processing the Ratings
# 4. Splitting the Dataset
# 5. Building the Model
# 6. Compile and Train the Model
# 7. Make Recommendations
# 8. top10 recommendation


# 1. Initial Imports
import tensorflow as tf
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import get_file

# 2. Loading the Data
URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
movielens_path = get_file("movielens.zip", URL, extract=True)

# 3. Processing the Data
with ZipFile(movielens_path) as z:
  with z.open("ml-latest-small/ratings.csv") as f:
    df = pd.read_csv(f)
    
#    - Processing User IDs
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
df["user"] = df["userId"].map(user2user_encoded)
num_users = len(user_encoded2user)

#    - Processing Movie IDs
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["movie"] = df["movieId"].map(movie2movie_encoded)
num_movies = len(movie_encoded2movie)

#    - Processing the Ratings
min, max = df["rating"].min(), df["rating"].max()

df["rating"] = df["rating"].apply(lambda x:(x-min)/(max-min))

# 4. Splitting the Dataset
X = df[["user", "movie"]].values
y = df["rating"].values
(x_train, x_val, y_train, y_val) = train_test_split(X, y,test_size=0.1,random_state=42)

print("Shape of the x_train: ", x_train.shape)
print("Shape of the y_train: ", y_train.shape)
print("Shape of the x_val: ", x_val.shape)
print("Shape of the x_val: ", y_val.shape)

# 5. Building the Model
# In TensorFlow, apart from Sequential API and Functional API, 
# there is a third option to build models: Model Subclassing. 


class RecommenderNet(tf.keras.Model):
  # __init function is to initialize the values of instance members for the new object
  def __init__(self, num_users, num_movies, embedding_size,**kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    # Variable for embedding size
    self.embedding_size = embedding_size
    # Variables for user count, and related weights and biases
    self.num_users = num_users
    self.user_embedding = Embedding(num_users,embedding_size,embeddings_initializer="he_normal",embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
    self.user_bias = Embedding(num_users, 1)
    # Variables for movie count, and related weights and biases
    self.num_movies = num_movies
    self.movie_embedding = Embedding(num_movies,embedding_size,embeddings_initializer="he_normal",embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
    self.movie_bias = Embedding(num_movies, 1)
    
  def call(self, inputs):
    # call function is for the dot products of user and movie vectors
    # It also accepts the inputs, feeds them into the layers,and feed into the final sigmoid layer
    # User vector and bias values with input values
    user_vector = self.user_embedding(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    # Movie vector and bias values with input values
    movie_vector = self.movie_embedding(inputs[:, 1])
    movie_bias = self.movie_bias(inputs[:, 1])
    # tf.tensordot calculcates the dot product
    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
    # Add all the components (including bias)
    x = dot_user_movie + user_bias + movie_bias
    # The sigmoid activation forces the rating to between 0 and 1
    return tf.nn.sigmoid(x)
  
# create an instance of this custom class to build our custom RecommenderNet model  
model = RecommenderNet(num_users, num_movies, embedding_size=50)

# 6. Compile and Train the Model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=0.001))

history = model.fit(x=x_train,y=y_train,batch_size=64,epochs=5,verbose=1,validation_data=(x_val, y_val))
                    
# 7. Make Recommendations
# selecting user_id
user_id = df.userId.sample(1).iloc[0]
print("The selected user ID is: ", user_id)

# checking and listing user not watching
movies_watched = df[df.userId == user_id]
not_watched = df[~df['movieId'].isin(movies_watched.movieId.values)]['movieId'].unique()
not_watched = [[movie2movie_encoded.get(x)] for x in not_watched]

print('The number of movies the user has not seen before: ',len(not_watched))

# generate the predicted movie ratings
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(([[user_encoder]] * len(not_watched), not_watched ))
ratings = model.predict(user_movie_array).flatten()

# 8. top10 recommendation
# selecting top10 with order of ratings
top10_indices = ratings.argsort()[-10:][::-1]

# get the movie ids for the top10
recommended_movie_ids = [movie_encoded2movie.get(not_watched[x][0]) for x in top10_indices]

# Create a DataFrame from Movies.csv file
with ZipFile(movielens_path) as z:
  with z.open("ml-latest-small/movies.csv") as f:
    movie_df = pd.read_csv(f)
    
top_movies_user = (movies_watched.sort_values(by="rating", ascending=False).head(10).movieId.values)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
print("Movies with high ratings from user")
print(movie_df_rows[['title','genres']])

print("--------------------------------------------------------------")
recommended_movies = movie_df[movie_df["movieId"].
isin(recommended_movie_ids)]
print("Top 10 movie recommendations")
print(recommended_movies[['title','genres']])
