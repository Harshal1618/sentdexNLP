import nltk

words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

#porter stemmer
porter = nltk.PorterStemmer()
stemWords = [porter.stem(word) for word in words]
print(stemWords)

text = "It is very important to be pythonly while you are pythoning with python.All pythoners have pythonded alleast once."
#tokenising
textTokens    = nltk.word_tokenize(text)
textStemWords = [porter.stem(word) for word in textTokens]
print(textStemWords)