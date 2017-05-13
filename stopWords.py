import nltk

sentence = "This is an example showing off stop word filtratrion."

#stopword
stopWords = nltk.corpus.stopwords.words("english")

print(stopWords)
print(len(stopWords))

#tokenising
wordTokens = nltk.word_tokenize(sentence)
filteredSentence = []

for word in wordTokens:
    if word not in stopWords:
        filteredSentence.append(word)

print(filteredSentence)

#could be done as this
filteredSentenceAgain = [word for word in wordTokens if word not in stopWords]

print(filteredSentenceAgain)