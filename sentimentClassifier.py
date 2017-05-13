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


#For testing actually
documents = [(list(nltk.corpus.movie_reviews.words(fileid)), category)
             for category in nltk.corpus.movie_reviews.categories()
             for fileid in nltk.corpus.movie_reviews.fileids(category)]

random.shuffle(documents)

# print(documents[:1])


#creating wordset from corpus movie_reviews
wordSet = nltk.corpus.movie_reviews.words()

allWords = [word.lower() for word in wordSet if word.isalpha()]


#stop words
stopWords = nltk.corpus.stopwords.words("english")

#Removing stop words from allWords
for word in stopWords:
    if word in allWords:
        allWords.remove(word)

#Frequency Distribution
allWordsFd = nltk.FreqDist(allWords)

#most frequent used words
# print(allWordsFd.most_common(15))

#frequency of 'stupid' word
# print(allWordsFd['stupid'])

#we will train only on first 4000 words
wordFeatures = list(allWordsFd.keys())[:4000]


def findFeatures(document):
    words = set(document)
    features = {}
    for word in wordFeatures:
        features[word] = (word in words)

    return features

# print(findFeatures(nltk.corpus.movie_reviews.words('neg/cv000_29416.txt')))

featureSet = [(findFeatures(review), category) for (review, category) in documents]

traningSet = featureSet[:1900]
testingSet = featureSet[1900:]

'''
    All Classifier used here , we have used default parameter of respective classifier
'''

#Naive Bayes Classifier
naiveBayesclassifier = nltk.NaiveBayesClassifier.train(traningSet)

'''
    loading previously trained classifier from the pickle
'''

# classifierFile = open('naivebayes.pickle', 'rb')
# classifier = pickle.load(classifierFile)
# classifierFile.close()

#classifier Accuracy
print("Original Naive Bayes Algorithm Accuracy :", nltk.classify.accuracy(naiveBayesclassifier, testingSet))

naiveBayesclassifier.show_most_informative_features(15)

'''
    Pickle is the way we save python objects
    so, we can use that save object
'''
#wb refers to write in bytes
saveClassifier = open("naivebayes.pickle", 'wb')
pickle.dump(naiveBayesclassifier, saveClassifier)
saveClassifier.close()

#Multi Nominal Naive Bayes Classifier
'''
    NLTK library is basically for Natural language processing. it is not for machine learning
    But NLTK library has got many classifier of its own like SklearnClassifier, Naive Bayes Classifier
    whereas
    Sklearn library is for Machine learning which has also classifier like Multinomial Naive Bayes, Gaussian Naive Bayes etc
    we could convert Sklearn Classifier into NLTK Sklearn Classifier.conversion allows to use the method of classifier(like train as NLTK classifier) of nltk library

'''
multiNomialNBClassifier = nltk.classify.scikitlearn.SklearnClassifier(MultinomialNB()) #initiliasing the MultinomialNB classifier and converting it into NLTK sklearn classifier
multiNomialNBClassifier.train(traningSet)

print("Original MultinomialNB Classifier Accuracy :", nltk.classify.accuracy(multiNomialNBClassifier, testingSet))

#Gaussian Naive Bayes Classifier
'''Throws Error while doing as rest i.e converting it into nltk Sklearn Classifier and train it'''
# gaussianNBClassifier = nltk.classify.scikitlearn.SklearnClassifier(GaussianNB()) #initiliasing the GuasianNB classifier and converting it into NLTK Sklearn classifier
# gaussianNBClassifier.train(traningSet)
#
# print("Original GaussianNB Classifier Accuracy :", nltk.classify.accuracy(gaussianNBClassifier, testingSet))

#Bernoulli Naive Bayes Classifier
bernoulliNBClassifier = nltk.classify.scikitlearn.SklearnClassifier(BernoulliNB()) #initiliasing the BernoulliNB classifier and converting it into NLTK Sklearn classifier
bernoulliNBClassifier.train(traningSet)

print("Original BernoulliNB Classifier Accuracy :", nltk.classify.accuracy(bernoulliNBClassifier, testingSet))

#Logistic Regression Classifier
logisticRegressionClassifier = nltk.classify.scikitlearn.SklearnClassifier(LogisticRegression())#initiliasing the Logistic Regression classifier and converting it into NLTK Sklearn classifier
logisticRegressionClassifier.train(traningSet)

print("Original Logistice Regression Classifier Accuracy", nltk.classify.accuracy(logisticRegressionClassifier, testingSet))

#Stochastic Gradient Decent Classifier
sgdClassifer = nltk.classify.scikitlearn.SklearnClassifier(SGDClassifier()) #initiliasing the SGDC classifier and converting it into NLTK Sklearn classifier
sgdClassifer.train(traningSet)

print("Original Stochastic Gradient Decent Classifier Accuracy", nltk.classify.accuracy(sgdClassifer, testingSet))

#Support Vector Classifier
svcClassifer = nltk.classify.scikitlearn.SklearnClassifier(SVC())#initiliasing the SVC classifier and converting it into NLTK Sklearn classifier
svcClassifer.train(traningSet)

print("Original SVC Classifier Accuracy :", nltk.classify.accuracy(svcClassifer, testingSet))

#Linear SVC Classifier
linearSVCClassier = nltk.classify.scikitlearn.SklearnClassifier(LinearSVC())#initiliasing the Linear SVC classifier and converting it into NLTK Sklearn classifier
linearSVCClassier.train(traningSet)

print("Original Linear SVC Accuracy :", nltk.classify.accuracy(linearSVCClassier, testingSet))

#NuSVC Classifier
nuSVCClassifier = nltk.classify.scikitlearn.SklearnClassifier(NuSVC())#initiliasing the NusSVC classifier and converting it into NLTK Sklearn classifier
nuSVCClassifier.train(traningSet)

print("Original NuSVC Classifier Accuracy :", nltk.classify.accuracy(nuSVCClassifier, traningSet))



#Our Custom Classifier
votedClassifier = VoteClassifier(naiveBayesclassifier, multiNomialNBClassifier, bernoulliNBClassifier, logisticRegressionClassifier, sgdClassifer, linearSVCClassier, nuSVCClassifier)

print("Voted Classifier Accuracy :", nltk.classify.accuracy(votedClassifier, testingSet))

print("Classification", votedClassifier.classify(testingSet[0][0]), "Confidence :", votedClassifier.confidence(testingSet[0][0]))
print("Classification", votedClassifier.classify(testingSet[1][0]), "Confidence :", votedClassifier.confidence(testingSet[1][0]))
print("Classification", votedClassifier.classify(testingSet[2][0]), "Confidence :", votedClassifier.confidence(testingSet[2][0]))
print("Classification", votedClassifier.classify(testingSet[3][0]), "Confidence :", votedClassifier.confidence(testingSet[3][0]))
print("Classification", votedClassifier.classify(testingSet[4][0]), "Confidence :", votedClassifier.confidence(testingSet[4][0]))
print("Classification", votedClassifier.classify(testingSet[5][0]), "Confidence :", votedClassifier.confidence(testingSet[5][0]))