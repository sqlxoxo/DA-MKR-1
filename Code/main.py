import os
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Завантаження стоп-слів
nltk.download('stopwords')

# Функція для завантаження документів
def load_documents(directory):
   documents = []
   filenames = []
   for filename in os.listdir(directory):
      if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
               documents.append(file.read())
               filenames.append(filename)
   return documents, filenames

# Завантажте документи з папки
documents, filenames = load_documents('./txtFiles')  # Змініть на шлях до вашої папки з файлами

# Обробка тексту
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True)
X = vectorizer.fit_transform(documents)

# Кластеризація з KMeans
kmeans = KMeans(n_clusters=2, random_state=42)  # Змініть на потрібну кількість кластерів
kmeans.fit(X)

# Виведення результатів кластеризації
clusters = kmeans.labels_
for file, cluster in zip(filenames, clusters):
   print(f"File: {file} -> Cluster: {cluster}")

# Візуалізація результатів
plt.scatter(X.toarray()[:, 0], X.toarray()[:, 1], c=clusters)
plt.title('K-Means Clustering')
plt.show()
