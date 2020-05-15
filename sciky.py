import sys
import csv
import re
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem.porter import *
from collections import defaultdict
from nltk.corpus import wordnet as wn, stopwords
# nltk.download('averaged_perceptron_tagger')
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.text import TextCollection
from collections import defaultdict
import string
from nltk.probability import FreqDist
import pickle
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')
# nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# text = file.read()

# def text_words(sent,comment_words):
#     for val in sent:
#
#         # typecaste each val to string
#         val = str(val)
#
#         val = re.sub("[^a-zA-Z]", " ", val)
#         # split the value
#         print(val)
#         tokens = val.split()
#
#         # Converts each token into lowercase
#         for i in range(len(tokens)):
#             tokens[i] = tokens[i].lower()
#
#             # print(tokens[i])
#             comment_words += " ".join(tokens) + " "
#
# def text_to_words(raw_text,stopwords):
#
#     # remove useless content
#     letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
#     # lowercase
#     words = letters_only.lower()
#     #remove stopwords
#     meaningful_words = [w for w in words if not w in stopwords]
#     return " ".join(meaningful_words)

    #stem
    #     token_words=word_tokenize(sentence)
    #     token_words
    #     stem_sentence=[]
    #     for word in token_words:
    #         stem_sentence.append(PorterStemmer(word))
    #         stem_sentence.append(" ")
    #     return "".join(stem_sentence)

# def tokenize(text):
#     stem = nltk.stem.SnowballStemmer('english')
#     text = text.lower()
#
#     for token in nltk.word_tokenize(text):
#         if token in string.punctuation: continue
#         yield stem.stem(token)
#
# def vectorize(corpus):
#     corpus = [nltk.tokenize(doc) for doc in corpus]
#     texts  = TextCollection(corpus)
#
#     for doc in corpus:
#         yield {
#             term: texts.tf_idf(term, doc)
#             for term in doc
#         }



# def vectorize(doc):
#     features = defaultdict(int)
#     for token in nltk.tokenize(doc):
#         features[token] += 1
#     return features


#
# with open('wine.csv', encoding="utf8", newline="") as csvfile:
# data = pd.read_csv('wine.csv')
data = load_files(r".\wine.csv")
stopwords_txt =  set(open('stopwords.txt').read().split())
# docs = data.description
# text = " ".join(sent for sent in docs)
stopwords = set(stopwords.words('english'))
# stopwords.update(stopwords_txt)


stemmer = WordNetLemmatizer()
corpus = data.description
documents = []
for doc in corpus:
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(doc))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)
    # print("doc",doc)


vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords)
X = vectorizer.fit_transform(documents).toarray()
print(documents)

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# for sent in corpus:
#
#     # TODO IF I NEED sentente ["addsadada","other sent","sentences 3 -----"]
#     tokenized_text=sent_tokenize(sent)
#     print(tokenized_text,"\n")
#     for w in tokenized_text:
#         if w not in stopwords:
#             filtered_sent.append(w)
#
#     #word tokenize to calculate frequencies
# print(filtered_sent)
#
# for sentence in filtered_sent:
#     tokenized_word = word_tokenize(sent)
#     fdist = FreqDist(tokenized_word)
#     print(fdist.most_common(5))


set = defaultdict(list)

i = 0



#
# print(text)



print("\n", "\n")
print(len(set), set["Jacquart NV Brut Mosaïque  (Champagne)"])

