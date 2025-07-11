import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
import random


# Load dataset from CSV file
df = pd.read_csv("movies.csv")
 #movie is removed since it is low quality and contains almost all genres, making it unhelpful for reccomendations.
df = df.drop(df[df.title == 'The Helix... Loaded'].index)

# Define which columns to combine for similarity calculation
features = ['keywords', 'cast', 'genres', 'director', 'overview']

# Fill any missing values in these columns with empty strings
for feature in features:
    df[feature] = df[feature].fillna('')


# Combine selected features into one text field for each movie
def combine_features(row):
    return " ".join([row[feature] for feature in features])


df["combined_features"] = df.apply(combine_features, axis=1)


# Create count matrix and cosine similarity matrix for all movies
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df["combined_features"])
cos_sim = cosine_similarity(count_matrix)


# Creating multiple lists before hand
rated_yes = []      
rated_no = []       
rating_scores = {}  
shown_movies = set()
skipped_movies = [] 
iterations = 0      
max_iterations = 14 
min_iterations = 5  


last_answer = 'yes' # Track last user answer
left_early = False  # Whether user quit early


# Start main loop to ask user about movies
while iterations < max_iterations:
    print(f"\nIteration {iterations + 1}.")


    # For first few iterations, pick random movie from first 60 rows
    if iterations < 4:
        first60 = list(set(range(60)) - shown_movies)
        if first60:
            movie_idx = random.choice(first60)
        else:
            break  # No unseen movies left in first 60
    else:
        if last_answer == 'yes':
            # If user liked last movie, find a similar movie
            similar_movies = list(enumerate(cos_sim[movie_idx]))
            sorted_similar = sorted(similar_movies, key=lambda x: x[1], reverse=True)
            for element in sorted_similar[1:]:  # Skip itself
                if element[0] not in shown_movies:
                    movie_idx = element[0]
                    break
        elif last_answer == 'no':
            # If user disliked last movie, pick a random unrelated movie
            unrelated = list(set(range(len(df))) - shown_movies)
            if unrelated:
                movie_idx = random.choice(unrelated)
            else:
                break  # No more unseen movies


    # Get movie title and set available options
    movie_title = df.iloc[movie_idx]['title']
    option_text = "(yes/no/na)" if iterations < min_iterations else "(yes/no/na/quit)"


    # Ask user for input repeatedly until valid response
    while True:
        print(f"Do you like the movie: '{movie_title}'? {option_text}")
        user_input = input("Your answer: ").strip().lower()


        # Allow quitting only if minimum iterations met and enough YES answers
        if user_input == 'quit' and iterations >= min_iterations and len(rated_yes) >= 7:
            print("\nYou chose to quit.")
            left_early = True
            break


        elif user_input == 'na':
            # User chose to skip this movie
            print("Movie skipped.")
            skipped_movies.append(movie_idx)
            shown_movies.add(movie_idx)
            break


        elif user_input == 'yes':
            # User liked the movie
            rated_yes.append(movie_idx)
            last_answer = 'yes'
            rating = random.uniform(6.0, 9.5)  # Assign hidden high rating
            shown_movies.add(movie_idx)
            rating_scores[movie_idx] = round(rating, 1)
            iterations += 1
            break


        elif user_input == 'no':
            # User disliked the movie
            rated_no.append(movie_idx)
            last_answer = 'no'
            rating = random.uniform(1.0, 4.5)  # Assign hidden low rating
            shown_movies.add(movie_idx)
            rating_scores[movie_idx] = round(rating, 1)
            iterations += 1
            break


        else:
            # Invalid input, prompt again
            print("Invalid Input. Please Type One of the Following: yes/no/na/quit.")


    if left_early:
        break


    # If iterations hit max but not enough YES, force to keep going
    if iterations >= max_iterations and len(rated_yes) < 7:
        max_iterations += 1


# Prepare recommendations: ensure at least one from top director
recommendations = []
if rated_yes:
    # Find the director of the user's highest-rated liked movie
    highest_rated_movie = max(rated_yes, key=lambda idx: rating_scores[idx])
    top_director = df.iloc[highest_rated_movie]['director']
    director_movies = df[df['director'] == top_director].index.tolist()
    # Recommend a movie from this director that user hasn't seen or rated
    for idx in director_movies:
        if idx not in shown_movies and idx not in rated_yes and idx not in rated_no:
            recommendations.append(df.iloc[idx]['title'])
            break


# Add more similar movies to fill up to 10 recommendations
scores = [0] * len(df)
for yes_movie in rated_yes:
    sim_scores = cos_sim[yes_movie]
    scores = [sum(x) for x in zip(scores, sim_scores)]


scored = list(enumerate(scores))
scored = sorted(scored, key=lambda x: x[1], reverse=True)


for idx, score in scored:
    if idx not in shown_movies and idx not in rated_yes and idx not in rated_no:
        recommendations.append(df.iloc[idx]['title'])
    if len(recommendations) >= 10:
        break


# Show skipped movies if user did not leave early
if not left_early and skipped_movies:
    print("\n\033[95mMovies you skipped (NA) that you may like:\033[0m")
    for i, idx in enumerate(skipped_movies[:10], start=1):
        print(f"{i}. {df.iloc[idx]['title']}")


# Show final recommendations
print("\n\033[36mRecommended movies for you:\033[0m")
for i, rec in enumerate(recommendations, start=1):
    print(f"{i}. {rec}")


## Calculate MAE (Mean Absolute Error) for hidden ratings vs similarity prediction
#---------------------------------------------------------------------------------------------------------
true_ratings = []
pred_ratings = []


for midx, true_rating in rating_scores.items():
    # Predict rating based on similarity to other liked movies
    if rated_yes:
        similarities = [cos_sim[midx][yes_idx] for yes_idx in rated_yes if midx != yes_idx]
        if similarities:
            pred_score = sum(similarities) / len(similarities)
            pred_rating = 1 + pred_score * 9  # Scale to 1-10
        else:
            pred_rating = 5.0  # Neutral fallback if no similarities
    else:
        pred_rating = 5.0
    true_ratings.append(true_rating)
    pred_ratings.append(pred_rating)


# Print session MAE if any ratings exist
if true_ratings:
    mae = mean_absolute_error(true_ratings, pred_ratings)
    print(f"\nSession MAE: {round(mae, 2)}")
#---------------------------------------------------------------------------------------------------------