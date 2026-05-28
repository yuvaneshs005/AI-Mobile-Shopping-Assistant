# AI-Mobile-Shopping-Assistant


AI Mobile Shopping Assistant is an NLP-based smartphone recommendation web application built using Flask, HTML, and CSS.

The system allows users to search for smartphones using natural language instead of traditional filters. Queries are processed using spaCy and Word2Vec Skip-gram embeddings, while cosine similarity ranks smartphones according to semantic relevance.

Examples:

* Samsung phone under 25000 with good camera
* Gaming phone with strong battery
* Best phone for photography
* 5G phone with good performance

---

## Features

* Natural language smartphone search
* Top-5 ranked recommendations
* Semantic matching using NLP
* Word2Vec Skip-gram embeddings
* Cosine similarity ranking
* Recommendation explanations
* Best Match highlight
* Smartphone comparison page
* Responsive shopping-style UI

---

## Tech Stack

### Backend

* Python
* Flask

### Frontend

* HTML
* CSS

### NLP & Recommendation

* spaCy
* Word2Vec (Skip-gram using Gensim)
* Cosine Similarity

---

## Project Workflow

1. User enters a natural language smartphone query.
2. spaCy preprocesses the text using:

   * Tokenization
   * Stopword removal
   * Lemmatization
3. Smartphone specifications are converted into semantic representations using Word2Vec Skip-gram.
4. Cosine similarity compares the user query with smartphone vectors.
5. Smartphones are ranked by similarity score.
6. Top-5 recommendations are displayed with explanations.

---

## Project Structure

```text
ai-mobile-shopping-assistant/
│
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

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## How Recommendations Are Generated

The recommendation engine combines smartphone specifications into searchable text representations.

The system:

* Preprocesses text using spaCy
* Trains a Word2Vec Skip-gram model
* Converts phones and queries into vectors
* Uses cosine similarity for ranking
* Returns the Top-5 most relevant smartphones

Recommendation explanations are generated based on semantic matching and smartphone specifications.

---

## Pages

### Home Page

Natural language search interface with example queries.

### Results Page

Displays Top-5 smartphone recommendations with:

* Similarity score
* Specifications
* Recommendation reasons
* Best Match badge

### Compare Page

Allows side-by-side comparison of two smartphones using:

* Price
* Battery
* Camera
* RAM
* Storage
* Processor
* Rating

