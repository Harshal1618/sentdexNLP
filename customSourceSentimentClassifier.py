#Gives wrong result for sentiments. Yet to check code. Follow sentdex-customSentimentClassifier or sentdexSentimentClassifier

import nltk
import random
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier #Stochastic Gradient Descent (SGD)
from sklearn.svm import SVC, LinearSVC, NuSVC #Support Vector Classification (SVC)

import statistics
'''
    Our Own custom classifier to improve the reliabilty and accuracy
    We are gonna use all above classifier except SVC Classifier since it's accuracy is usually less than 50%
'''
class VoteClassifier(nltk.classify.ClassifierI):
    # *classifiers represents any number of parameters
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    #overide method
    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)
        return statistics.mode(votes)


    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        choiceVotes = votes.count(statistics.mode(votes))
        confidenceLevel = choiceVotes/ len(votes)

        return confidenceLevel

#loading from custom sources
positiveReviews = open('reviews/positive.txt', 'r').read()
negativeReviews = open('reviews/negative.txt', 'r').read()

documents = []

for review in positiveReviews.split("\n"):
    documents.append((review, 'pos'))

for review in negativeReviews.split("\n"):
    documents.append((review, "neg"))

random.shuffle(documents)

#creating wordset
posWordSet = nltk.word_tokenize(positiveReviews)
negWordSet = nltk.word_tokenize(negativeReviews)

'''
    loading allWords from pickle file
'''
allWordsFile = open('pickles/allWords.pickle', 'rb')
allWords     = pickle.load(allWordsFile)
allWordsFile.close()

# allWords = []
'''
    we are only interested in following types of word for
    sentiment analysis like adjective, adverb and verb

    so pos_tag returns pos tag list as ('remain', 'VBP')
    considering first letter of pos representation

    J is adjective, R is adverb and V is verb

'''

# allowedWordTypes = ["J", "R", "V"]
#
# posWordNorm = [word.lower() for word in posWordSet if word.isalpha()]
# negWordNorm = [word.lower() for word in negWordSet if word.isalpha()]
#
# posWordTags = nltk.pos_tag(posWordNorm)
# negWordTags = nltk.pos_tag(negWordNorm)
#
# for word,tag in posWordTags:
#     if tag[0] in allowedWordTypes:
#         allWords.append(word)
#
# for word,tag in negWordTags:
#     if tag[0] in allowedWordTypes:
#         allWords.append(word)

'''
    Pickling
'''
# saveClassifier = open('pickles/allWords.pickle', 'wb')
# pickle.dump(allWords, saveClassifier)
# saveClassifier.close()


#stop words
stopWords = nltk.corpus.stopwords.words("english")

#Removing stop words from allWords
for word in stopWords:
    if word in allWords:
        allWords.remove(word)

#Frequency Distribution
# allWordsFd = nltk.FreqDist(allWords)

'''
    loading allWordsFd from pickle file
'''
allWordsFdFile = open('pickles/frequency-distribution.pickle', 'rb')
allWordsFd     = pickle.load(allWordsFdFile)
allWordsFdFile.close()

'''
    Pickling
'''
# saveClassifier = open('pickles/frequency-distribution.pickle', 'wb')
# pickle.dump(allWordsFd, saveClassifier)
# saveClassifier.close()

#we will train only on first 6000 words
wordFeatures = list(allWordsFd.keys())[:6000]


def findFeatures(document):
    words = set(nltk.word_tokenize(document))
    features = {}
    for word in wordFeatures:
        features[word] = (word in words)

    return features


# featureSet = [(findFeatures(review), category) for (review, category) in documents]
'''
    loading featureSet from pickle file
'''
featureSetFile = open('pickles/feature-set.pickle', 'rb')
featureSet     = pickle.load(featureSetFile)
featureSetFile.close()

'''
    Pickling
'''
# saveClassifier = open('pickles/feature-set.pickle', 'wb')
# pickle.dump(featureSet, saveClassifier)
# saveClassifier.close()


traningSet = featureSet[:10000]
testingSet = featureSet[10000:]

'''
    All Classifier used here , we have used default parameter of respective classifier
'''

#Naive Bayes Classifier
# naiveBayesclassifier = nltk.NaiveBayesClassifier.train(traningSet)

'''
    loading previously trained Naive Bayes Classifier from the pickle
'''

naivebayesClassifierFile = open('pickles/naivebayesclassifier.pickle', 'rb')
naiveBayesclassifier = pickle.load(naivebayesClassifierFile)
naivebayesClassifierFile.close()

#classifier Accuracy
print("Original Naive Bayes Algorithm Accuracy :", nltk.classify.accuracy(naiveBayesclassifier, testingSet))

naiveBayesclassifier.show_most_informative_features(15)

'''
    Pickle is the way we save python objects
    so, we can use that save object
'''
#wb refers to write in bytes
# saveClassifier = open("pickles/naivebayesclassifier.pickle", 'wb')
# pickle.dump(naiveBayesclassifier, saveClassifier)
# saveClassifier.close()

#Multi Nominal Naive Bayes Classifier
'''
    NLTK library is basically for Natural language processing. it is not for machine learning
    But NLTK library has got many classifier of its own like SklearnClassifier, Naive Bayes Classifier
    whereas
    Sklearn library is for Machine learning which has also classifier like Multinomial Naive Bayes, Gaussian Naive Bayes etc
    we could convert Sklearn Classifier into NLTK Sklearn Classifier.conversion allows to use the method of classifier(like train as NLTK classifier) of nltk library

'''
# multiNomialNBClassifier = nltk.classify.scikitlearn.SklearnClassifier(MultinomialNB()) #initiliasing the MultinomialNB classifier and converting it into NLTK sklearn classifier
# multiNomialNBClassifier.train(traningSet)

'''
    loading previously trained MultinomialNB Classifier from the pickle
'''

multiNomialNBClassifierFile = open('pickles/multinomial-classifier.pickle', 'rb')
multiNomialNBClassifier = pickle.load(multiNomialNBClassifierFile)
multiNomialNBClassifierFile.close()

print("Original MultinomialNB Classifier Accuracy :", nltk.classify.accuracy(multiNomialNBClassifier, testingSet))


'''
    Pickling
'''
# saveClassifier = open('pickles/multinomial-classifier.pickle', 'wb')
# pickle.dump(multiNomialNBClassifier, saveClassifier)
# saveClassifier.close()

#Gaussian Naive Bayes Classifier
'''Throws Error while doing as rest i.e converting it into nltk Sklearn Classifier and train it'''
# gaussianNBClassifier = nltk.classify.scikitlearn.SklearnClassifier(GaussianNB()) #initiliasing the GuasianNB classifier and converting it into NLTK Sklearn classifier
# gaussianNBClassifier.train(traningSet)
#
# print("Original GaussianNB Classifier Accuracy :", nltk.classify.accuracy(gaussianNBClassifier, testingSet))

#Bernoulli Naive Bayes Classifier

# bernoulliNBClassifier = nltk.classify.scikitlearn.SklearnClassifier(BernoulliNB()) #initiliasing the BernoulliNB classifier and converting it into NLTK Sklearn classifier
# bernoulliNBClassifier.train(traningSet)

'''
    loading previously trained BernoulliNB Classifier from the pickle
'''

bernoulliNBClassifierFile = open('pickles/bernoulli-classifier.pickle', 'rb')
bernoulliNBClassifier = pickle.load(bernoulliNBClassifierFile)
bernoulliNBClassifierFile.close()


print("Original BernoulliNB Classifier Accuracy :", nltk.classify.accuracy(bernoulliNBClassifier, testingSet))


'''
    Pickling
'''
# saveClassifier = open('pickles/bernoulli-classifier.pickle', 'wb')
# pickle.dump(bernoulliNBClassifier, saveClassifier)
# saveClassifier.close()

#Logistic Regression Classifier

# logisticRegressionClassifier = nltk.classify.scikitlearn.SklearnClassifier(LogisticRegression())#initiliasing the Logistic Regression classifier and converting it into NLTK Sklearn classifier
# logisticRegressionClassifier.train(traningSet)

'''
    loading previously trained Logistic Regression Classifier from the pickle
'''

logisticRegressionClassifierFile = open('pickles/logistic-regression.pickle', 'rb')
logisticRegressionClassifier = pickle.load(logisticRegressionClassifierFile)
logisticRegressionClassifierFile.close()

print("Original Logistic Regression Classifier Accuracy", nltk.classify.accuracy(logisticRegressionClassifier, testingSet))


'''
    Pickling
'''
# saveClassifier = open('pickles/logistic-regression.pickle', 'wb')
# pickle.dump(logisticRegressionClassifier, saveClassifier)
# saveClassifier.close()

#Stochastic Gradient Decent Classifier

'''
    loading previously trained Stochastic Gradient Decent Classifier from the pickle
'''

sgdClassiferFile = open('pickles/sgd-classifier.pickle', 'rb')
sgdClassifer = pickle.load(sgdClassiferFile)
sgdClassiferFile.close()

# sgdClassifer = nltk.classify.scikitlearn.SklearnClassifier(SGDClassifier()) #initiliasing the SGDC classifier and converting it into NLTK Sklearn classifier
# sgdClassifer.train(traningSet)

print("Original Stochastic Gradient Decent Classifier Accuracy", nltk.classify.accuracy(sgdClassifer, testingSet))


'''
    Pickling
'''
# saveClassifier = open('pickles/sgd-classifier.pickle', 'wb')
# pickle.dump(sgdClassifer, saveClassifier)
# saveClassifier.close()

#Support Vector Classifier

# svcClassifer = nltk.classify.scikitlearn.SklearnClassifier(SVC())#initiliasing the SVC classifier and converting it into NLTK Sklearn classifier
# svcClassifer.train(traningSet)

'''
    loading previously trained SVC Classifier from the pickle
'''

svcClassiferFile = open('pickles/svc-classifier.pickle', 'rb')
svcClassifer = pickle.load(svcClassiferFile)
svcClassiferFile.close()

print("Original SVC Classifier Accuracy :", nltk.classify.accuracy(svcClassifer, testingSet))


'''
    Pickling
'''
# saveClassifier = open('pickles/svc-classifier.pickle', 'wb')
# pickle.dump(svcClassifer, saveClassifier)
# saveClassifier.close()

#Linear SVC Classifier

# linearSVCClassier = nltk.classify.scikitlearn.SklearnClassifier(LinearSVC())#initiliasing the Linear SVC classifier and converting it into NLTK Sklearn classifier
# linearSVCClassier.train(traningSet)

'''
    loading previously trained Linear SVC Classifier from the pickle
'''

linearSVCClassierFile = open('pickles/linearSVC-classifier.pickle', 'rb')
linearSVCClassier = pickle.load(linearSVCClassierFile)
linearSVCClassierFile.close()


print("Original Linear SVC Accuracy :", nltk.classify.accuracy(linearSVCClassier, testingSet))


'''
    Pickling
'''
# saveClassifier = open('pickles/linearSVC-classifier.pickle', 'wb')
# pickle.dump(linearSVCClassier, saveClassifier)
# saveClassifier.close()


#NuSVC Classifier

# nuSVCClassifier = nltk.classify.scikitlearn.SklearnClassifier(NuSVC()) #initiliasing the NusSVC classifier and converting it into NLTK Sklearn classifier
# nuSVCClassifier.train(traningSet)

'''
    loading previously trained NuSVC Classifier from the pickle
'''

# nuSVCClassifierFile = open('pickles/nuSVC-classifier.pickle', 'rb')
# nuSVCClassifier = pickle.load(nuSVCClassifierFile)
# nuSVCClassifierFile.close()
#
#
# print("Original NuSVC Classifier Accuracy :", nltk.classify.accuracy(nuSVCClassifier, traningSet))


'''
    Pickling
'''
# saveClassifier = open('pickles/nuSVC-classifier.pickle', 'wb')
# pickle.dump(nuSVCClassifier, saveClassifier)
# saveClassifier.close()


#Our Custom Classifier
# votedClassifier = VoteClassifier(naiveBayesclassifier, multiNomialNBClassifier, bernoulliNBClassifier, logisticRegressionClassifier, sgdClassifer, linearSVCClassier)
votedClassifier = VoteClassifier(
                                  naiveBayesclassifier,
                                  linearSVCClassier,
                                  multiNomialNBClassifier,
                                  bernoulliNBClassifier,
                                  logisticRegressionClassifier)


print("Voted Classifier Accuracy :", nltk.classify.accuracy(votedClassifier, testingSet))

# print("Classification", votedClassifier.classify(testingSet[0][0]), "Confidence :", votedClassifier.confidence(testingSet[0][0]))
# print("Classification", votedClassifier.classify(testingSet[1][0]), "Confidence :", votedClassifier.confidence(testingSet[1][0]))
# print("Classification", votedClassifier.classify(testingSet[2][0]), "Confidence :", votedClassifier.confidence(testingSet[2][0]))
# print("Classification", votedClassifier.classify(testingSet[3][0]), "Confidence :", votedClassifier.confidence(testingSet[3][0]))
# print("Classification", votedClassifier.classify(testingSet[4][0]), "Confidence :", votedClassifier.confidence(testingSet[4][0]))
# print("Classification", votedClassifier.classify(testingSet[5][0]), "Confidence :", votedClassifier.confidence(testingSet[5][0]))

def sentiment(text):
    features = findFeatures(text)

    return votedClassifier.classify(features), votedClassifier.confidence(features)