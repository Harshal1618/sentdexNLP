import nltk

#loading raw bible text
bibleText = nltk.corpus.gutenberg.raw("bible-kjv.txt")

#sentence tokenizing
bibleSentTokens = nltk.sent_tokenize(bibleText)

print(bibleSentTokens[5:15])
print("\n")

#creating wordset from bible text
bibleWordSet = nltk.corpus.gutenberg.words("bible-kjv.txt")


print(bibleWordSet[:10])

#creating sentence set from bible
bibleSentSet = nltk.corpus.gutenberg.sents("bible-kjv.txt")
print(bibleSentTokens[:10])