import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
from fuzzywuzzy import process
import random

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.warning("SpaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load and preprocess the CSV file
file_path = 'data.csv'
df = pd.read_csv(file_path)

# Fill NaN values in Level 1 and Level 2
df['Level 1: Categories'].ffill(inplace=True)
df['Level 2: Subcategories'].ffill(inplace=True)

# Split Level 3 into multiple rows
df = df.set_index(['Level 1: Categories', 'Level 2: Subcategories'])['Level 3: Detailed Subcategories'].str.split(', ', expand=True).stack().reset_index()
df.columns = ['Level 1', 'Level 2', 'level_3', 'Level 3']
df = df.drop('level_3', axis=1)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# TF-IDF Matrix
tfidf_matrix = vectorizer.fit_transform(df.apply(lambda x: ' '.join(x), axis=1))

# Initialize Nearest Neighbors model
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(tfidf_matrix)

def extract_keywords(input_text):
    doc = nlp(input_text)
    # Extract nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return keywords

def find_similar(input_text, n=5):
    # Extract keywords from input_text
    keywords = extract_keywords(input_text)
    
    # Find closest matching terms in 'Level 1', 'Level 2', 'Level 3'
    matched_terms = []
    for keyword in keywords:
        level1_matches = process.extractOne(keyword, df['Level 1'])
        level2_matches = process.extractOne(keyword, df['Level 2'])
        level3_matches = process.extractOne(keyword, df['Level 3'])
        
        if level1_matches and level1_matches[1] >= 80:
            matched_terms.append(df[df['Level 1'] == level1_matches[0]])
        elif level2_matches and level2_matches[1] >= 80:
            matched_terms.append(df[df['Level 2'] == level2_matches[0]])
        elif level3_matches and level3_matches[1] >= 80:
            matched_terms.append(df[df['Level 3'] == level3_matches[0]])
    
    if not matched_terms:
        return "No matches found."
    
    # Concatenate matched terms into a single string for vectorization
    matched_text = ' '.join([' '.join(term.iloc[0].values) for term in matched_terms])
    
    # Vectorize the matched_text
    input_vector = vectorizer.transform([matched_text])
    
    # Find nearest neighbors
    distances, indices = nbrs.kneighbors(input_vector, n_neighbors=n)
    
    # Get the top similar entry
    top_entry = df.iloc[indices[0][0]]
    
    # Format the result using varied templates
    templates = [
        f"Based on your query '{input_text}', I believe the most relevant category for you is found within the textile taxonomy. It falls under the main category '{top_entry['Level 1']}', which is further specified into the subcategory '{top_entry['Level 2']}'. Finally, within this subcategory, you will find the detailed category '{top_entry['Level 3']}'. This should precisely match your needs.",
        
        f"Your search for '{input_text}' has led me to identify a specific path within our textile classification. The primary category is '{top_entry['Level 1']}', narrowing down to '{top_entry['Level 2']}', and ultimately leading to '{top_entry['Level 3']}'. This detailed path should be exactly what you're looking for in the textile domain.",
        
        f"Considering your input '{input_text}', the best match within our taxonomy starts at the top level with '{top_entry['Level 1']}'. This broad category then branches into '{top_entry['Level 2']}', and within this subset, the most detailed and relevant category is '{top_entry['Level 3']}'. I hope this helps in your search.",
        
        f"Upon analyzing your query '{input_text}', it seems that the closest match within the textile taxonomy hierarchy begins with '{top_entry['Level 1']}' as the main category. This is further divided into the subcategory '{top_entry['Level 2']}', with the most specific match being '{top_entry['Level 3']}'. This structured path should guide you accurately."
    ]
    
    response = random.choice(templates)
    
    return response

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-container {
        max-width: 800px;
        margin: auto;
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        color: #555555;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .text-input {
        width: 100%;
        padding: 10px;
        font-size: 16px;
        margin-bottom: 20px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .button {
        display: block;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        border-radius: 5px;
        margin: auto;
    }
    .button:hover {
        background-color: #45a049;
    }
    .response {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        font-size: 16px;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">Textile Taxonomy Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Find the best match for your textile query</div>', unsafe_allow_html=True)

input_text = st.text_input("Query", "", key="query", help="Type your query here", max_chars=100)
if st.button('Find'):
    if input_text:
        response = find_similar(input_text, n=1)
        st.markdown(f'<div class="response">{response}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a query.")

st.markdown('</div>', unsafe_allow_html=True)
