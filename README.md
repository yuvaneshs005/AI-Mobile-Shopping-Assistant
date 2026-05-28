# AI Mobile Shopping Assistant

An NLP-based smartphone recommendation website built with Python Flask, HTML, and CSS.

The app accepts natural language searches such as:

- Samsung phone under 25000 with good camera
- Gaming phone with strong battery
- Best phone for photography
- 5G phone with good performance

It recommends the Top 5 smartphones using spaCy preprocessing, Word2Vec Skip-gram embeddings, and cosine similarity.

## Project Structure

```text
ai-mobile-shopping-assistant/
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
├── data/
│   └── smartphones.csv
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── results.html
    └── compare.html
```

## Setup Instructions

1. Open a terminal in this folder.

2. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Run the Flask app.

```bash
python app.py
```

5. Open the local website.

```text
http://127.0.0.1:5000
```

## How The Recommendation System Works

### Why spaCy is used

spaCy is used for NLP preprocessing. It tokenizes user queries and phone descriptions, converts text to lowercase, removes stopwords, and lemmatizes words. This makes words like "gaming", "games", and "game" easier to compare.

### How Word2Vec Skip-gram works

Word2Vec learns word meanings from nearby words. This project uses the Skip-gram method, where the model learns to predict surrounding context words from a target word. For example, words near "battery" may include "5000mah", "charging", and "backup", helping the model learn semantic relationships.

### Why cosine similarity is used

Cosine similarity measures how close two vectors are in direction. It is useful because recommendations should depend on meaning, not only exact matching. A user query vector is compared with each phone vector, and phones with higher similarity are ranked higher.

### How recommendations are generated

1. The dataset fields are combined into a single smartphone description.
2. spaCy preprocesses every description.
3. Gensim trains a Word2Vec Skip-gram model on the descriptions.
4. Each phone is represented by the average of its word vectors.
5. The user query is preprocessed and converted into a vector.
6. Cosine similarity ranks all smartphones.
7. The app displays the Top 5 recommendations with simple explanations.

## Main Features

- Professional homepage with natural language search
- Top 5 smartphone recommendations
- Best Match badge for the highest ranked result
- Similarity score for each phone
- Explanation for every recommendation
- Phone comparison page
- Responsive shopping-style UI

## File-by-File Explanation

### `app.py`

Creates the Flask application and defines the website routes:

- `/` shows the homepage
- `/recommend` processes user queries and displays recommendations
- `/compare` compares two selected phones

### `recommender.py`

Contains the NLP and recommendation logic:

- Loads the CSV dataset
- Preprocesses text using spaCy
- Trains Word2Vec with Skip-gram
- Creates phone and query vectors
- Ranks phones using cosine similarity
- Generates recommendation explanations

### `templates/base.html`

Common layout used by all pages. It includes the navigation bar and CSS file.

### `templates/index.html`

Homepage with project title, description, search box, and sample queries.

### `templates/results.html`

Displays the Top 5 recommended smartphones as cards with specs, score, and explanations.

### `templates/compare.html`

Allows the user to select two phones and compare price, battery, camera, RAM, storage, processor, and rating.

### `static/style.css`

Contains the complete responsive styling for the premium shopping assistant interface.

## Note

The app includes a simple fallback so pages can still load before spaCy and Gensim are installed. For the correct college project implementation, install all packages from `requirements.txt` and download `en_core_web_sm`.
