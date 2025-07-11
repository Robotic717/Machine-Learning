# Movie Recommendation System

This Python script is a **simple interactive movie recommender** built with **pandas** and **scikit-learn**.  
It asks you about different movies, learns what you like, and suggests similar movies based on your preferences.

---

## Features

Loads a dataset of movies from a `movies.csv` file.  
Uses **_content-based filtering_** with keywords, cast, genres, director, and overview.  
Calculates similarity using **_cosine similarity_** between movies using a bag-of-words representation.  
Interactive loop:
- Asks if you like/dislike a movie, or if you have watched it.
- Picks the next movie based on your answers.
- Allows skipping or quitting (after a minimum number of answers).
Ensures you see at least one movie from your favorite director.  
Generates **_10 personalized recommendations_**.  
Calculates **_Mean Absolute Error (MAE)_** to evaluate how well the system’s similarity predictions match your hidden ratings.

---

## How It Works

1. **Load and clean data**  
   - Reads `movies.csv`.  

2. **Combine text features**  
   - Merges important columns (`keywords`, `cast`, `genres`, `...`) into a single text field.

3. **Create a similarity matrix**  
   - Uses `CountVectorizer` to create a term frequency matrix.
   - Computes pairwise cosine similarity.

4. **Interactive session**  
   - Shows you a random movie.
   - If you like it, the next movie is similar.
   - If you don’t, the next is random which is not similar.
   - You can skip movies or quit after minimum rounds.

5. **Generate recommendations**  
   - Picks more movies from your favorite director.
   - Adds the highest scoring similar movies.
   - Ensures you get 10 final recommendations.

6. **Show skipped movies**  
   If you didn’t quit early, the system also lists skipped titles you might like.

7. **Calculate session MAE**  
   Compares your hidden preference scores to similarity predictions.

---

## Requirements

- Python 3.7+
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

OR Install dependencies with:

```bash
pip install pandas scikit-learn
