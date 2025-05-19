import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega o arquivo CSV
df = pd.read_csv('C:/Users/lab53/Downloads/restaurants_reviews.csv')

# Verifica o nome da coluna com texto — substitua por outro nome se necessário
text_column = 'review_text'  # supondo que a coluna se chame 'review'
if text_column not in df.columns:
    print("Colunas disponíveis:", df.columns)
    raise ValueError("Ajuste o nome da coluna com as avaliações de texto.")

# Preenche valores ausentes com string vazia
texts = df[text_column].fillna("")

# Vetoriza os textos usando TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Cria vetor para o termo "food"
term = "food"
term_vector = vectorizer.transform([term])

# Calcula similaridade do cosseno entre "food" e todos os documentos
cosine_similarities = cosine_similarity(term_vector, tfidf_matrix).flatten()

# Adiciona a similaridade ao DataFrame original
df['cosine_similarity_with_food'] = cosine_similarities

# Exibe os 5 textos mais semelhantes ao termo "food"
top_matches = df.sort_values(by='cosine_similarity_with_food', ascending=False).head(5)

print(top_matches[[text_column, 'cosine_similarity_with_food']])

