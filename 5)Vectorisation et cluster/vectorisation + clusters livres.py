import streamlit as st
import os
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# Étape 1 : Chargement des données
txt_files = [
    "LORD_OF_THE_RING.txt",
    "FALL_OF_GONDOLIN.txt",
    "CHILDREN_OF_HURIN.txt",
    "SILMARILLION.txt",
    "UNFINISHED_TALES_OF_NUMENOR_AND_MIDDLE_EARTH.txt",
    "LETTERS.txt",
    "THE_HOBBIT.txt",
    "FARMER_GILS_OF_HAM.txt",
    "LETTERS_FROM_FATHER_CHRISTMAS.txt",
    "MR_BLISS.txt",
    "SMITH_OF_WOOTTON_MAJOR.txt",
    "THE_FELLOWSHIP_OF_THE_RING.txt",
    "THE_HOMECOMING_OF_BEORHTNOTH_BEORHTHELMS_SON.txt",
    "THE_LEGEND_OF_SIGURD_AND_GUDRUN.txt",
    "THE_RETURN_OF_THE_KING.txt",
    "THE_TWO_TOWERS.txt",
]
book_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in txt_files]

corpus = []
for file_name in txt_files:
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
        corpus.append(text)

# Prétraitement des données
nltk.download('stopwords')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

# Initialisation du lemmatiseur
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return words

preprocessed_corpus = [preprocess_text(text) for text in corpus]

# Entraînement de Word2Vec
model_w2v = Word2Vec(preprocessed_corpus, vector_size=100, window=5, min_count=1, sg=0)

def get_text_vector_w2v(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

book_vectors_w2v = [get_text_vector_w2v(text, model_w2v) for text in preprocessed_corpus]

# Entraînement du modèle SentenceTransformer
model_sbert = SentenceTransformer('bert-base-nli-mean-tokens')
book_vectors_sbert = model_sbert.encode(corpus)

# Réduction dimensionnelle pour visualisation 2D
pca_w2v_2d = PCA(n_components=2)
book_vectors_w2v_2d = pca_w2v_2d.fit_transform(book_vectors_w2v)

pca_sbert_2d = PCA(n_components=2)
book_vectors_sbert_2d = pca_sbert_2d.fit_transform(book_vectors_sbert)

# Fonction pour déterminer le nombre optimal de clusters
def optimal_cluster_number(vectors):
    scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(vectors)
        score = silhouette_score(vectors, kmeans.labels_)
        scores.append(score)

    optimal_k = range(2, 10)[scores.index(max(scores))]
    return optimal_k

n_clusters_w2v = optimal_cluster_number(book_vectors_w2v_2d)
n_clusters_sbert = optimal_cluster_number(book_vectors_sbert_2d)

# Modification de la fonction de visualisation
def plot_vectors_2d(vectors, labels, title, cluster_labels):
    plt.figure(figsize=(10, 10))
    colors = list(mcolors.TABLEAU_COLORS.keys())
    number_labels = range(len(labels))

    for i, num_label in enumerate(number_labels):
        plt.scatter(vectors[i, 0], vectors[i, 1], color=colors[cluster_labels[i]])
        plt.text(vectors[i, 0], vectors[i, 1], str(num_label), color='black', fontsize=8)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.grid(True)
    st.pyplot()

# Création d'une légende associant numéros et titres
def create_legend(book_names):
    legend = "Légende:\n" + "\n".join([f"{i}: {name}" for i, name in enumerate(book_names)])
    return legend

# Appel de la fonction pour créer la légende
legend = create_legend(book_names)

# Affichage dans Streamlit
st.title('Visualisation des Vecteurs de Livres')

# Affichage de la légende
st.text(legend)

# Clusterisation des vecteurs
cluster_labels_w2v = KMeans(n_clusters=n_clusters_w2v, random_state=0).fit_predict(book_vectors_w2v_2d)
cluster_labels_sbert = KMeans(n_clusters=n_clusters_sbert, random_state=0).fit_predict(book_vectors_sbert_2d)

# Noms des livres
book_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in txt_files]

# Affichage des graphiques avec les modifications
st.subheader('Visualisation en 2D avec Word2Vec et Clustering')
plot_vectors_2d(book_vectors_w2v_2d, book_names, 'Word2Vec avec Clustering', cluster_labels_w2v)

st.subheader('Visualisation en 2D avec SentenceTransformer (BERT) et Clustering')
plot_vectors_2d(book_vectors_sbert_2d, book_names, 'SentenceTransformer (BERT) avec Clustering', cluster_labels_sbert)

# Charger le fichier CSV des labels. Assurez-vous que le fichier est dans le même répertoire que votre script.
try:
    label_data = pd.read_csv('tolkien_books_labels.csv')
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier CSV: {e}")

# Nettoyer les données de labels
label_data = label_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Encoder les labels
le = LabelEncoder()
for column in label_data.columns[1:]:  # Ignorer la colonne 'Titre'
    label_data[column] = le.fit_transform(label_data[column].astype(str))

# Séparer les caractéristiques et les labels
X = np.array(book_vectors_sbert)
y = label_data.drop('Titre', axis=1)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Affichage dans Streamlit
st.title('Arbres de Décision pour Chaque Label')

# Entraînement et visualisation des arbres de décision pour chaque label
for label in y.columns:
    st.subheader(f'Arbre de Décision pour {label}')
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train[label])
    
    # Calcul du score de précision
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test[label], y_pred)
    st.write(f'Score de Précision pour {label}: {score:.2f}')
    
    # Visualisation de l'arbre de décision
    fig, ax = plt.subplots(figsize=(20, 10))
    # Convertir les noms de classe en liste
    class_names = le.classes_.tolist()  # Modification ici
    plot_tree(clf, filled=True, feature_names=[f'Feature_{i}' for i in range(X_train.shape[1])], class_names=class_names, ax=ax)
    st.pyplot(fig)