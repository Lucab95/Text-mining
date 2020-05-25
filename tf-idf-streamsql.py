import csv
import re
from collections import defaultdict
import math
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

file = open("wine.csv",encoding="utf8")
reader = csv.reader(file)

stopwords_txt =  set(open('stopwords.txt').read().split())
# docs = data.description
# text = " ".join(sent for sent in docs)
stopwords = set(stopwords.words('english'))
stopwords.update(stopwords_txt)




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

def computeCountDict():
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

def computeIDFDict():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in count:
        idfDict[word] = math.log(len(data) / count[word])
    return idfDict



data=[]
for row in reader:
    row[2] = re.sub(r'[^\w\s]', '', row[2])
    sent_in_word = []
    for word in row[2].lower().split():
        if word not in stopwords:
                sent_in_word.append(word)
    data.append(sent_in_word)

data = data[1:]
print(data[0])

#Removes header

# print(data)
tfDict = []
print("tfDict done")
for i,sent in enumerate(data):
    tfDict.append(computeReviewTFDict(sent))
    # print(len(tfDict), sent)
    # if i ==150:
    #     break

# print(tfDict[3])
count={}
print("count phase..")
count.update(computeCountDict())

count = {k: v for k, v in sorted(count.items(), key=lambda item: -item[1])}
print(count)
idfDict = computeIDFDict()
# idfDict['fruit']

img = WordCloud(width = 800, height = 800,background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate_from_frequencies(count)
plt.imshow(img, interpolation='bilinear')
plt.savefig("img/worldcloud.png", format="png")
plt.axis("off")
plt.show()