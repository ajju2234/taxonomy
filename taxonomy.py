import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
from fuzzywuzzy import process

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

# Initialize spaCy for keyword extraction
nlp = spacy.load('en_core_web_sm')

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
    
    # Get similar entries
    similar_entries = df.iloc[indices[0]]
    
    return similar_entries

# Example usage
input_text = "i want Knitted Fabrics"
similar_entries = find_similar(input_text, n=3)
print(similar_entries)
