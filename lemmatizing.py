import nltk

lemmatizer = nltk.WordNetLemmatizer()

print(lemmatizer.lemmatize("Cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))

print(lemmatizer.lemmatize("better", pos = "a")) #default parameter of pos(part of speech) is noun
print(lemmatizer.lemmatize("better"))
