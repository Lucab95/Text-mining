import csv
import re
import logging as log, sys

from collections import defaultdict
from copy import deepcopy
import pickle
from pathlib import Path
import config
import createDir as dir
import math
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from tfIdf import computeReviewTFDict, computeCountDict, computeIDFDict
from nltk.stem import WordNetLemmatizer

# import test
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
spacy.prefer_gpu()

###########################     Main    ###########################
if config.general['debug']:
    log.getLogger("matplotlib").setLevel(log.WARNING)
    log.basicConfig(stream=sys.stderr, level=log.DEBUG)


#open file and create stopwords set
file = open(config.general['dataset_path'],encoding="utf8")
stopwords_txt =  set(open(config.general['stopwords']).read().split())
stopwords = set(stopwords.words('english'))
stopwords.update(stopwords_txt)

pickle_corpus = Path(config.general["pickle_folder"]) / "corpus.pickle"
pickle_split_sent = Path(config.general["pickle_folder"]) / "split_sent.pickle"
pickle_tfDict = Path(config.general["pickle_folder"]) / "pickle_tfDict.pickle"

# try to load the pickels
corpus = dir.picke_load(pickle_corpus)
split_sent = dir.picke_load(pickle_split_sent)
tfDict = dir.picke_load(pickle_tfDict)

# corpus      = []
# split_sent  = []
# tfDict      = []

# if there are no previous data, create and save them
if len(corpus)==0  or len(split_sent)== 0 or len(tfDict)== 0:
    print("Create corpus..")
    dir.create_dir(config.general["pickle_folder"])
    reader = csv.reader(file)
    for row in reader:
        corpus.append(row[2])
    # remove first row -> description
    corpus = corpus[1:]

    #create split sent and
    print("Create single word docs..")
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for doc in corpus:
        doc = re.sub(r'[^\w\s\d-]', '', doc) #remove spec chars and punctuation except dash
        sent_word_split = []
        for word in doc.lower().split():
            if word not in stopwords:
                    sent_word_split.append(word)
        split_sent.append(sent_word_split)
    print("calculate TF in dict")
    for i, sent in enumerate(split_sent):
        tfDict.append(computeReviewTFDict(sent))

    pickle.dump(corpus, open(pickle_corpus, "wb"))
    pickle.dump(split_sent, open(pickle_split_sent, "wb"))
    pickle.dump(tfDict, open(pickle_tfDict, "wb"))
else:
    print("Load data from pickle..")


"""stanford ner"""
#stanford ner tagger
# jar = './stanford-ner-tagger/stanford-ner.jar'
# model = './stanford-ner-tagger/ner-model-english.ser.gz'
# st = StanfordNERTagger(model, jar)
i=0
# set= set()
lemmatizer = WordNetLemmatizer()


with open('wine-flavour.txt', 'w') as f:

    for doc in corpus:
        #     print("word", word)
        #     set.add(word)

        # print (sent)
        #
        if i>10:
            f.write("%s\n" % doc)
            log.debug("doc% s" % doc)
            log.debug("valore di i %s" % str(i))
        i = i+1
        if i==100:
            break
    # for s in set:
    #     # print(set)
    #     print(s)
    #     f.write("%s\n" % s)
    #     # # print(word)

nlp = spacy.load("model")
i=0

if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe("ner")

x=False
if x:
    for doc in corpus:
        print(doc)
        doc = nlp(doc)
        cleaned = [y for y in doc
                   if not y.is_stop and y.pos_ != 'PUNCT']
        print("cleaned",cleaned)
        raw = [(x.lemma_, x.pos_) for x in cleaned]
        ents = [(x.text, x.label_) for x in doc.ents]
        print("ents",ents)
        print("later",raw)

        i = i+1
        if i==10:
            break




"""TF IDF"""

print("number of docs: ",len(corpus))
print("number of split_sent:", len(split_sent))


tfDict_ordered = {}
print("count phase and order..")
tfDict_ordered.update(computeCountDict(split_sent))
tfDict_count = {k: v for k, v in sorted(tfDict_ordered.items(), key=lambda item: -item[1])}
print("tfDict result", tfDict_count)
print("compute IDF")
idfDict = computeIDFDict(tfDict_ordered,corpus)
# idfDict['fruit']

img = WordCloud(width = 1000, height = 1000,background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate_from_frequencies(tfDict_ordered)


plt.imshow(img, interpolation='bilinear')
plt.savefig("img/worldcloud.png", format="png")
plt.axis("off")
plt.show()