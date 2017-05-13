import nltk

#loading George W Bush Speech
trainText  = nltk.corpus.state_union.raw("2005-GWBush.txt")
sampleText = nltk.corpus.state_union.raw("2006-GWBush.txt")

#Punkt Sentence Tokeniser
punktSentenceTokenizer = nltk.PunktSentenceTokenizer(trainText)

sentTokens = punktSentenceTokenizer.tokenize(sampleText)

def processContent():
    try:
        for tokens in sentTokens:
            wordTokens = nltk.word_tokenize(tokens)
            tagged = nltk.pos_tag(wordTokens)
            chunkGram = '''Chunk: {<.*>+}
                                  }<VB.?|IN|DT|TO>+{''' #chunk everything except chink like VB
            chunkParser = nltk.RegexpParser(chunkGram)

            chunked = chunkParser.parse(tagged)

            print(chunked)

    except Exception as e:
        print(str(e))

processContent()