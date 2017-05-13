import nltk

text = "Hello there, How are you doing today ? The weather is great and Python is awesome.The sky is pinkish-blue.You shouldn't eat cardboard"

#sentence Tokens
sentenceTokens = nltk.sent_tokenize(text)
print(sentenceTokens)

#word tokens
wordTokens = nltk.word_tokenize(text)
print(wordTokens)