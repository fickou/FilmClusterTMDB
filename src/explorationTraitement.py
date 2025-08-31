
# 1. Importation des librairies et du dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import TfidfVectorizer



# 2. Chargement  des données
df = pd.read_csv("../donnees/tmdb_5000_movies.csv")


# Aperçu des colonnes du dataset
columns = df.columns.tolist()
columns


#nombre de lignes et de colonnes
shape = df.shape
shape


# Langues distinctes dans original_language
langues_originales = df['original_language'].unique()
print("Langues distinctes dans original_language :", langues_originales)
nb_langues = df['original_language'].nunique()
print(f"Nombre de langues distinctes dans original_language : {nb_langues}")




# Analyse exploratoire du dataset TMDB


#1. aperçu des données
print(f"Nombre de films : {len(df)}")
print(df.info())
print(df.describe())


#2. Statistiques descriptives des variables numériques
print(df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']].describe())



#3. Visualisation des distributions
#3.1 Distribution du budget
plt.figure(figsize=(10,5))
sns.histplot(df['budget'].replace(0, pd.NA).dropna(), bins=50, kde=True)
plt.title("Distribution des budgets (excluant les 0)")
plt.xlabel("Budget")
plt.ylabel("Nombre de films")
plt.show()



#3.2 Distribution des notes moyennes
plt.figure(figsize=(8,4))
sns.histplot(df['vote_average'], bins=20, kde=True)
plt.title("Distribution des notes moyennes")
plt.xlabel("Note moyenne")
plt.ylabel("Nombre de films")
plt.show()



#3. Extraction et analyse des genres
# Parsing des genres JSON
df['genres_list'] = df['genres'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)] if pd.notnull(x) else [])

# Comptage total des genres
from collections import Counter
all_genres = [genre for sublist in df['genres_list'] for genre in sublist]
genre_counts = Counter(all_genres)

# Affichage des genres les plus fréquents
plt.figure(figsize=(12,6))
sns.barplot(x=list(genre_counts.keys()), y=list(genre_counts.values()))
plt.xticks(rotation=45)
plt.title("Répartition des genres de films")
plt.xlabel("Genres")
plt.ylabel("Nombre de films")
plt.show()



#5. Analyse des sorties par année
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

movies_per_year = df.groupby('release_year').size()
plt.figure(figsize=(12,6))
movies_per_year.plot()
plt.title('Nombre de films sortis par année')
plt.xlabel('Année')
plt.ylabel('Nombre de films')
plt.show()



# Nettoyage des données (valeurs manquantes, doublons)


# 2. Gestion des valeurs manquantes


# a. Variables numériques (budget, revenue, runtime)


from sklearn.impute import SimpleImputer

# Remplacement des valeurs 0 par NaN pour les colonnes numériques

df['budget'] = df['budget'].replace(0, np.nan)
df['revenue'] = df['revenue'].replace(0, np.nan)
df['runtime'] = df['runtime'].replace(0, np.nan)

imputer = SimpleImputer(strategy='median')

# Imputation des valeurs manquantes
df[['budget', 'revenue', 'runtime']] = imputer.fit_transform(df[['budget', 'revenue', 'runtime']])


# b. Variables textuelles (overview, genres)


avant = len(df)
df = df.dropna(subset=['overview'])
apres = len(df)
print(f"Lignes sans synopsis supprimées : {avant - apres}")

df.to_csv("../donnees/tmdb_5000_movies_nettoyer.csv", index=False)


# 3. Vérification finale du nettoyage


print("Valeurs manquantes après nettoyage :")
print(df.isnull().sum())



import matplotlib.pyplot as plt
import seaborn as sns

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
plt.figure(figsize=(8,2))
sns.barplot(x=missing.index, y=missing.values)
plt.title('Valeurs manquantes par colonne')
plt.xticks(rotation=45)
plt.ylabel('Nombre de valeurs manquantes')
plt.show()



# Prétraitement textuel des synopsis (nettoyage, tokenization, suppression stop words)


#  Importation des librairies necessaires


import re
import nltk
from nltk.tokenize import word_tokenize
import stopwordsiso as stopwords
import sys


# 1. Nettoyage du texte


def nettoyage_texte(texte):
    
    if not isinstance(texte, str) or pd.isna(texte):
        return ''
    texte = texte.lower()
    texte = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', texte)
    return texte


# 2. La tokenization


def tokenization_texte(texte):
    
    return word_tokenize(texte)


# 3. Suppression stop words


def suppression_stopwords(tokens, langue):
   
    if langue and stopwords.has_lang(langue):
        stop_words = stopwords.stopwords(langue)
    else:
        stop_words = set()
    return [mot for mot in tokens if mot not in stop_words]





# 4. Pretraitement


def pretraitement_complet(texte, langue):
    
    if not isinstance(texte, str) or not texte.strip():
        return ''
    texte_nettoye = nettoyage_texte(texte)
    tokens = tokenization_texte(texte_nettoye)
    tokens_filtrés = suppression_stopwords(tokens, langue)
    return ' '.join(tokens_filtrés)


# Fonction de détection de langue


from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def detecter_langue_texte(texte):
    try:
        if pd.isna(texte) or not isinstance(texte, str) or texte.strip() == "":
            return "inconnu"
        return detect(texte)  
    except Exception as e:
        print(f"Erreur pour : {str(texte)[:30]}... -> {e}")
        return "inconnu"


from deep_translator import GoogleTranslator
from tqdm import tqdm


# Fonction de traduction avec gestion d’erreur et barre de progression
def traduire_texte(texte):
    try:
        if pd.isna(texte) or texte.strip() == '':
            return ''
        return GoogleTranslator(source='auto', target='fr').translate(texte)
    except Exception as e:
        print(f"Erreur de traduction pour : {texte[:30]}... : {e}")
        return ''
tqdm.pandas()
df['overview_fr'] = df['overview'].progress_apply(traduire_texte)


df['overview_fr'].to_csv("../donnees/overview_fr.csv", index=False, encoding="utf-8")



# Détection de la langue du synopsis


overview_fr = pd.read_csv("../donnees/overview_fr.csv")


df['overview_fr'] = overview_fr['overview_fr']
df = df.dropna(subset=['overview_fr'])
df['langue_detectee'] = df['overview_fr'].apply(detecter_langue_texte)



# Identification des lignes où la langue détectée diffère de original_language


df['langue_differe'] = df.apply(
    lambda row: row['langue_detectee'] != row['original_language'] if pd.notnull(row['langue_detectee']) else False,
    axis=1
)


# Affichage de conflit de langue


print("Différences entre original_language et langue détectée dans le synopsis :")
print(df[df['langue_differe']][['original_language', 'langue_detectee', 'overview_fr']])


# ajout colonne langue effective pour le prétraitement :


df['langue_effective'] = df.apply(
    lambda row: row['langue_detectee'] if pd.notnull(row['langue_detectee']) else row['original_language'],
    axis=1
)


# 4. Application du prétraitement complet sur chaque synopsis


df['synopsis_nettoye'] = df.apply(
    lambda row: pretraitement_complet(row['overview_fr'], row['langue_effective']),
    axis=1
)


print(df[['original_language', 'langue_detectee', 'langue_effective', 'overview_fr', 'synopsis_nettoye']].head(5))


print(df.loc[4000:4007, ['original_language', 'langue_detectee', 'langue_effective', 'overview_fr', 'synopsis_nettoye']])


# Sauvegarde de la colonne 'synopsis_nettoye' dans un fichier texte


chemin_fichier = "../donnees/corpus_synopsis.txt"

with open(chemin_fichier, 'w', encoding='utf-8') as fichier:
    for texte in df['synopsis_nettoye'].fillna(''):
        fichier.write(texte + '\n')
        
print(f"Corpus sauvegardé dans {chemin_fichier}")



# Vectorisation textuelle (TF-IDF ou embeddings)
# 


# Vectorisation TF-IDF


from sklearn.feature_extraction.text import TfidfVectorizer


def vectoriser_tf_idf(corpus, max_termes=5000):
    vectoriseur = TfidfVectorizer(max_features=max_termes)
    matrice_tfidf = vectoriseur.fit_transform(corpus)
    print(f"TF-IDF : matrice de forme {matrice_tfidf.shape}")
    return matrice_tfidf, vectoriseur


with open("../donnees/corpus_synopsis.txt", 'r', encoding='utf-8') as f:
    corpus = f.read().splitlines()


matrice_tfidf, vectoriseur = vectoriser_tf_idf(corpus)


# Conversion en DataFrame pandas pour affichage
df_tfidf = pd.DataFrame(matrice_tfidf.toarray(), columns=vectoriseur.get_feature_names_out())



# Affichage des premières lignes
print(df_tfidf.head())


# Pour voir les termes les plus représentatifs dans un document spécifique,
print(df_tfidf.loc[0].sort_values(ascending=False).head(10))


# Vectorisation embeddings Word2Vec par moyenne des vecteurs mots 


from gensim.models import KeyedVectors


# chargement du model


def charger_modele_word2vec():
    print("Chargement du modèle Word2Vec pré-entraîné...")
    modele = KeyedVectors.load("../donnees/word2vec-google-news-300.model", mmap='r')
    return modele


def vectoriser_par_embeddings(corpus, modele):
    vecteurs = []
    for doc in corpus:
        tokens = tokenization_texte(doc)
        vecteurs_mots = [modele[mot] for mot in tokens if mot in modele]
        if vecteurs_mots:
            vecteur_doc = np.mean(vecteurs_mots, axis=0)
        else:
            vecteur_doc = np.zeros(modele.vector_size)
        vecteurs.append(vecteur_doc)
    matrice_embeddings = np.array(vecteurs)
    print(f"Embeddings : matrice de forme {matrice_embeddings.shape}")
    return matrice_embeddings


modele_w2v = charger_modele_word2vec()


print("Calcul des embeddings...")
embeddings = vectoriser_par_embeddings(corpus, modele_w2v)


embeddings = pd.DataFrame(embeddings)
print(embeddings.head())


# Enregistrer
np.save("../donnees/embeddings_films.npy", embeddings)


# Encodage et normalisation des données numériques et catégorielles


from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Colonnes catégorielles et numériques à traiter
colonnes_categorielles = ['genres', 'original_language', 'production_countries']
colonnes_numeriques = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']


# --- Encodage des variables catégorielles ---
def encoder_categorie(df, colonnes):
    encodeur = OneHotEncoder(sparse_output=False, drop='first')
    donnees_a_encoder = df[colonnes].fillna('inconnu')
    valeurs_encodees = encodeur.fit_transform(donnees_a_encoder)
    noms_colonnes = encodeur.get_feature_names_out(colonnes)
    df_encode = pd.DataFrame(valeurs_encodees, columns=noms_colonnes, index=df.index)
    df_rest = df.drop(columns=colonnes)
    df_final = pd.concat([df_rest, df_encode], axis=1)
    return df_final


df_encoded = encoder_categorie(df, colonnes_categorielles)
df_encoded


# --- Normalisation des variables numériques ---
def normaliser_numerique(df, colonnes):
    normaliseur = StandardScaler()
    donnees_a_normaliser = df[colonnes].fillna(df[colonnes].mean())
    valeurs_normalisees = normaliseur.fit_transform(donnees_a_normaliser)
    df_normalise = pd.DataFrame(valeurs_normalisees, columns=colonnes, index=df.index)
    df_rest = df.drop(columns=colonnes)
    df_final = pd.concat([df_rest, df_normalise], axis=1)
    return df_final


df_prep = normaliser_numerique(df_encoded, colonnes_numeriques)
df_prep


# Fusion des données numériques + catégorielles et embeddings
df_prep = df_prep.reset_index(drop=True)

# Garder uniquement les colonnes numériques
df_numerique = df_prep.select_dtypes(include=[np.number])




if len(embeddings) < len(df_numerique):
    diff = len(df_numerique) - len(embeddings)
    zero_pad = np.zeros((diff, embeddings.shape[1]))
    embeddings = np.vstack([embeddings, zero_pad])


# Fusionner uniquement le numérique + embeddings
donnees_fusionnees = np.hstack([df_numerique.values, embeddings])


# Enregistrement dans un fichier .npy
donnees_fusionnees = np.array(donnees_fusionnees, dtype=np.float32)
np.save("../donnees/donnees_fusionnees.npy", donnees_fusionnees)


# Pour charger les donnees pour le clustering


donnees_fusionnees_chargees = np.load("../donnees/donnees_fusionnees.npy", allow_pickle=True)

print("Matrice rechargée, forme :", donnees_fusionnees_chargees.shape)
print(donnees_fusionnees_chargees)