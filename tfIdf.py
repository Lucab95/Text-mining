import math


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

def computeIDFDict(tfDict_ordered,corpus):
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in tfDict_ordered:
        idfDict[word] = math.log(len(corpus) / tfDict_ordered[word])
    return idfDict