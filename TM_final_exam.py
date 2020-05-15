import csv
import re
from collections import defaultdict
from copy import deepcopy
from nltk.tag.stanford import StanfordNERTagger
import pickle
from pathlib import Path
import config
import createDir as dir
import math
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def computeReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all
    the unique words in the review and whose values are their
    corresponding tf.
    """
    # Counts the number of times the word appears in review
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1
    # Computes tf for each word
    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict

def computeCountDict(tfDict):
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for review in tfDict:
        for word in review:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

def computeIDFDict(tfDict_ordered):
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in tfDict_ordered:
        idfDict[word] = math.log(len(corpus) / tfDict_ordered[word])
    return idfDict

###########################     Main    ###########################

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
    for doc in corpus:
        doc = re.sub(r'[^\w\s]', '', doc)
        sent_in_word = []
        for word in doc.lower().split():
            if word not in stopwords:
                    sent_in_word.append(word)
        split_sent.append(sent_in_word)

    print("calculate TF in dict")
    for i, sent in enumerate(split_sent):
        tfDict.append(computeReviewTFDict(sent))

    pickle.dump(corpus, open(pickle_corpus, "wb"))
    pickle.dump(split_sent, open(pickle_split_sent, "wb"))
    pickle.dump(tfDict, open(pickle_tfDict, "wb"))
else:
    print("Load data from pickle..")


print("number of docs: ",len(corpus))
print("number of split_sent:", len(split_sent))


tfDict_ordered = {}
print("count phase and order..")
tfDict_ordered.update(computeCountDict(split_sent))
tfDict_count = {k: v for k, v in sorted(tfDict_ordered.items(), key=lambda item: -item[1])}
print(tfDict_count)

print("compute IDF")
idfDict = computeIDFDict(tfDict_ordered)
# idfDict['fruit']

img = WordCloud(width = 800, height = 800,background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate_from_frequencies(tfDict_ordered)


plt.imshow(img, interpolation='bilinear')
plt.savefig("img/worldcloud.png", format="png")
plt.axis("off")
plt.show()