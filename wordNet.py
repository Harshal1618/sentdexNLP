import nltk

programSynonymSet = nltk.corpus.wordnet.synsets("program")

#synset
# print(programSynonymSet)
# print(programSynonymSet[0])
# print(programSynonymSet[0].name())
# print(programSynonymSet[0].lemmas())
# print(programSynonymSet[0].lemmas()[0])
# print(programSynonymSet[0].lemmas()[0].name())

#definition
# print(programSynonymSet[0].definition())
#
#examples
# print(programSynonymSet[0].examples())

synonyms = []
antonyms = []

for synonym in nltk.corpus.wordnet.synsets("good"):
    for lemma in synonym.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

#Semantic Similarity

word1 = nltk.corpus.wordnet.synset("ship.n.01")# noun and first one
word2 = nltk.corpus.wordnet.synset("boat.n.01")

print(word1.wup_similarity(word2)) #gives percentage

word3 = nltk.corpus.wordnet.synset("car.n.01")
word4 = nltk.corpus.wordnet.synset("cat.n.01")
word5 = nltk.corpus.wordnet.synset("cactus.n.01")

print(word1.wup_similarity(word3))
print(word1.wup_similarity(word4))
print(word1.wup_similarity(word5))



