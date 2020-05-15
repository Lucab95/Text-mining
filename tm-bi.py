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
from sklearn.feature_extraction.text import CountVectorizer

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
data = pd.read_csv('wine.csv')
stopwords_txt = set(open('stopwords.txt').read().split())
# docs = data.description
# text = " ".join(sent for sent in docs)
stopwords = set(stopwords.words('english'))
stopwords.update(stopwords_txt)

row = data.description
corpus = []
for doc in row:
   corpus.append(doc)

filtered_sent=[]
for sent in corpus:

    # TODO IF I NEED sentente ["addsadada","other sent","sentences 3 -----"]
    tokenized_text=sent_tokenize(sent)
    # print(tokenized_text,"\n")
    for w in tokenized_text:
        if w not in stopwords:
            filtered_sent.append(w)

    #word tokenize to calculate frequencies
print(filtered_sent)


for sentence in filtered_sent:
    tokenized_word = word_tokenize(sent)
    fdist = FreqDist(tokenized_word)
    # print(fdist.most_common(5))

# vector = vectorize(corpus)
# print(corpus)
# # vectors = map(vectorize, corpus)
# print(vectors)
text = ''
# text = text_words(data.description,text)


# for val in data.description:
#
#     # typecaste each val to string
#     val = str(val)
#
#     val = re.sub("[^a-zA-Z]", " ", val)
#     # split the value
#     print(val)
#     tokens = val.split()
#
#     # Converts each token into lowercase
#     for i in range(len(tokens)):
#         tokens[i] = tokens[i].lower()
#
#         # print(tokens[i])
#     comment_words += " ".join(tokens) + " "


# WordCloud.to_file('img/frequencies.png')
# text = ""
#create full clean text
# corpus = text_to_words(text,stopwords)
# for val in corpus:
#     print(corpus)
# print(corpus)
# stemming

set = defaultdict(list)

i = 0

# for row in tabin:
# sid = SentimentIntensityAnalyzer()
# score = sid.polarity_scores(row[2])
# print(score, "of sentence", row[2])

# corpus = nltk.sent_tokenize(docs)
# w_frequencies = {}
# for sentence in corpus:
#     tokens = nltk.word_tokenize(sentence)
#     for token in tokens:
#         if token not in w_frequencies.keys():
#             w_frequencies[token] = 1
#         else:
#             w_frequencies[token] += 1
# text = ""
# print(corpus)
# for corp in corpus:
#     # print(corp)
#     for w in corp:
#         # print(w)
#         text = text + w
#     if i > 10000:
#         break
#
#     i += 1
#         break
# text = " ".join(review for review in data.description)
# print(final)
# print ("There are {} words in the combination of all review.".format(len(final)))


print(text)
# Generate a word cloud image
img = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(fdist)
WordCloud.to_file('frequencies.png',img)
wordcloud = WordCloud().generate(text)


"""worldcloudddd"""
                # img = WordCloud(width = 800, height = 800,background_color ='black',
                #                 stopwords = stopwords,
                #                 min_font_size = 10).generate(text)
                # plt.imshow(img, interpolation='bilinear')
                # plt.savefig("img/spa_wine-bi.png", format="png")
                # plt.axis("off")
                # plt.show()





# for i in range(len(corpus)):
#     # corpus[i] = corpus[i].lower()
#     # corpus[i] = re.sub(r'\W', ' ', corpus[i])
#     # corpus[i] = re.sub(r'\s+', ' ', corpus[i])
#     print(len(corpus))
# print(row[2])
# words = word_tokenize(row[2])
# ps = PorterStemmer()
# for w in words:
#     rootWord = ps.stem(w)
#     # print(rootWord)

# # print(row[13])
# if "Jacquart NV Brut Mosaïque  (Champagne)" ==row[11]:
# #     print(i,row[2])
# set[row[11]].append(row[2])
# # if row[11]=='':
# #     empty += 1
# # print(i)
# i += 1

# #lemmatization
#   wordnet_lemmatizer = WordNetLemmatizer()
#   tokenization = word_tokenize(row[2])
#   tag_map = defaultdict(lambda: wn.NOUN)
#   tag_map['J'] = wn.ADJ
#   tag_map['V'] = wn.VERB
#   tag_map['R'] = wn.ADV
#   lemma_function = WordNetLemmatizer()
#   for token, tag in pos_tag(tokenization):
#       lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
#       print(token, "=>", lemma, "tag:",tag[0])
#
#   # for w in tokenization:
#   #     print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))
#   #     # doit=False
# print(len(corpus))


print("\n", "\n")
print(len(set), set["Jacquart NV Brut Mosaïque  (Champagne)"])
# for reviews in set:
#   #lemmatization
#   print(set[reviews])
#   for review in set[reviews]:
#
#
#       wordnet_lemmatizer = WordNetLemmatizer()
#       tokenization = word_tokenize(review)
#       tag_map = defaultdict(lambda: wn.NOUN)
#       tag_map['J'] = wn.ADJ
#       tag_map['V'] = wn.VERB
#       tag_map['R'] = wn.ADV
#       lemma_function = WordNetLemmatizer()
#       for token, tag in pos_tag(tokenization):
#           lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
#           # print(token, "=>", lemma, "tag:",tag[0])
#   empty+=1
#   if empty>9:
#       break
#           # for w in tokenization:
#     print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))
#     # doit=False
