import pandas as pd
import numpy as np


###Import the data
path = "ml-latest-small\{}.csv"

###Change into DF
movies = pd.read_csv(path.format("movies"), usecols=["movieId", "title"]
                     , dtype={"movieId": "int32", "title": "str"})
ratings = pd.read_csv(path.format("ratings")
                     , usecols=["userId", "movieId", "rating"]
                     , dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
#print(ratings.head())
##movies:  movieId | title
##ratings:  userId | movieId | rating


###DF of movie vs user
movie_user = ratings.pivot(index = "userId", columns = "movieId", values = "rating").fillna(0)
movie_user_array = movie_user.values
##movie_user:  movieId | 1~....
##                0    | 
##                1    | 

#print(movie_user_array)


print("Please input your userID.\n")
Uid = input()
Uid = int(Uid)


###SVD
## xij = uik * Skk * vkj
U, S, V = np.linalg.svd(movie_user_array)
sigma = np.diag(S)
##reduce the size of matrices 
U_r = U[:,:30]
sigma_r = sigma[:30,:30]
V_r = V[:,:30]
VT_r = V_r.T
#print(np.shape(VT_r))

US = np.dot(U_r, sigma_r)
ratings_prediction = np.dot(US, VT_r)
ratings_prediction_df = pd.DataFrame(ratings_prediction, columns = movie_user.columns)
#print(ratings_prediction_df)

###Search user data by UID
def search_user_data(ratings, Uid):
    user_data = ratings[ratings["userId"] == Uid]   ##Find user data from table ratings
    user_data_full = user_data.merge(movies)        ##Combine table user_data with table movies
    user_data_full_sorted = user_data_full.sort_values(["rating"], ascending = False)   ##Sort by rating

    return user_data_full_sorted

#print(search_user_data(ratings, Uid))

###Prediction to user
def predicted_user_data(ratings_prediction_df, Uid):
    Uid_row = Uid - 1   ##row number starts from 0, but Uid starts from 1
    predictions = ratings_prediction_df.iloc[Uid_row]   ##Find movieID rating by user which predicted by SVD
    predictions_df = pd.DataFrame(predictions).reset_index()
    predictions_df = predictions_df.rename(columns = {Uid_row: "Predicted_rating"})
    return predictions_df
    
#print(predicted_user_data(ratings_prediction_df, Uid)[:10])
    
def sorted_data(data, a, num_recommenadation):
    data_sorted = data.sort_values([a], ascending = False)   ##Sort values for column "a" 
    data_sorted = data_sorted.iloc[:num_recommenadation, :-1]   ##Only print "movieId" | "title" without "ratings"
    return data_sorted
    
def select_unrated_data(user_data_full_sorted, movies):
    rated_data = movies["movieId"].isin(user_data_full_sorted["movieId"])   ##"movieId" in [user_data_full_sorted] will be "True"
    unrated_data = ~rated_data   ##"~" makes "True" been "Fulse"
    
    return unrated_data
    
def recommendation(Uid, ratings, movies):
    pud = predicted_user_data(ratings_prediction_df, Uid)
    sud = select_unrated_data(search_user_data(ratings, Uid), movies)
    recommend = movies[sud].merge(pud)   ##Only contains unrated titles
    recommend.sorted = sorted_data(recommend, "Predicted_rating", 10)
    
    return recommend.sorted
    
f = open("recommendation_to_you.txt", "w")
print(recommendation(Uid, ratings, movies), file =f)
f.close()
    
    
    
    











