import sys
import csv
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from collections import defaultdict
from nltk.corpus import wordnet as wn
# nltk.download('averaged_perceptron_tagger')
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

                # def stemSentence(sentence):
                #     token_words=word_tokenize(sentence)
                #     token_words
                #     stem_sentence=[]
                #     for word in token_words:
                #         stem_sentence.append(PorterStemmer(word))
                #         stem_sentence.append(" ")
                #     return "".join(stem_sentence)

with open('wine.csv', encoding="utf8", newline="") as csvfile:
  tabin = csv.reader(csvfile)

  i=0

  set = defaultdict(list)
  empty = 0
  for row in tabin:

      set[row[11]].append(row[2])

      i += 1

  print("\n","\n")
  print(len(set),set["Jacquart NV Brut Mosaïque  (Champagne)"])
  for reviews in set:
    #lemmatization
    print(set[reviews])
    for review in set[reviews]:


        wordnet_lemmatizer = WordNetLemmatizer()
        tokenization = word_tokenize(review)
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        lemma_function = WordNetLemmatizer()
        for token, tag in pos_tag(tokenization):
            lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
            print(token, "=>", lemma, "tag:",tag[0])
    empty+=1
    if empty>9:
        break
