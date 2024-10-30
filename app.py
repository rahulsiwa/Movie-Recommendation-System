

from flask import Flask, render_template, request, flash, redirect
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['title'] + ' ' + x['director'] + ' ' + x['cast'] + ' ' + x['listed_in'] + ' ' + x['description']

def get_recommendations(title, cosine_sim):
    title = title.replace(' ', '').lower()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    result = netflix_overall['title'].iloc[movie_indices]
    result = result.to_frame()
    result = result.reset_index(drop=True)
    return result

# Load and prepare data
netflix_overall = pd.read_csv('netflix_titles.csv')
netflix_data = pd.read_csv('netflix_titles.csv').fillna('')
new_features = ['title', 'director', 'cast', 'listed_in', 'description']
netflix_data = netflix_data[new_features]

for feature in new_features:
    netflix_data[feature] = netflix_data[feature].apply(clean_data)

netflix_data['soup'] = netflix_data.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(netflix_data['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(netflix_data.index, index=netflix_data['title']).to_dict()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about', methods=['POST'])
def getvalue():
    moviename = request.form['moviename']
    result = get_recommendations(moviename, cosine_sim2)
    if result is None:
        flash("Sorry, the movie you're looking for is not in our dataset. Please try again with a different title.")
        return redirect('/')  # Redirect to the homepage with an error message
    return render_template('result.html', tables=[result.to_html(classes='data')], titles=result.columns.values)

if __name__ == '__main__':
    app.run(debug=True)  # Debug mode enabled for detailed error messages
