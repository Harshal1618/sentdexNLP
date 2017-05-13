import nltk

#loading speech of George W Bush
trainText  = nltk.corpus.state_union.raw("2005-GWBush.txt")
sampleText = nltk.corpus.state_union.raw("2006-GWBush.txt")

#Punkt Sentence Tokeniser
punktSentenceTokeniser = nltk.PunktSentenceTokenizer(trainText)

sentTokens = punktSentenceTokeniser.tokenize(sampleText)

def processContent():
    try:
        for tokens in sentTokens:
            wordTokens = nltk.word_tokenize(tokens)
            tagged = nltk.pos_tag(wordTokens)

            # nameEnt = nltk.ne_chunk(tagged , binary=True)#adding binary= True will not classify Name Entity as location , money or something else.it just list as Name Entity
            nameEnt = nltk.ne_chunk(tagged)
            nameEnt.draw()

    except Exception as e:
        print(str(e))

processContent()