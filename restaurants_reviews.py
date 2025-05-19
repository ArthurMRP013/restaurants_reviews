import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv('C:/Users/lab53/Downloads/restaurants_reviews.csv')
text_column = 'review_text'  # supondo que a coluna se chame 'review'
if text_column not in df.columns:
    print("Colunas disponíveis:", df.columns)
    raise ValueError("Ajuste o nome da coluna com as avaliações de texto.")
texts = df[text_column].fillna("")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
term = "food"
term_vector = vectorizer.transform([term])
cosine_similarities = cosine_similarity(term_vector, tfidf_matrix).flatten()
df['cosine_similarity_with_food'] = cosine_similarities
top_matches = df.sort_values(by='cosine_similarity_with_food', ascending=False).head(5)

print(top_matches[[text_column, 'cosine_similarity_with_food']])

