import nltk

#loading Geroge W Bush speech
trainText  = nltk.corpus.state_union.raw("2005-GWBush.txt")
sampleText = nltk.corpus.state_union.raw("2006-GWBush.txt")

#punkt sentence tokenizer
punktSentenceTokenizer = nltk.PunktSentenceTokenizer(trainText)

sentTokens = punktSentenceTokenizer.tokenize(sampleText)



def processContent():
    try:
        for token in sentTokens:
            wordTokens = nltk.word_tokenize(token)
            tagged = nltk.pos_tag(wordTokens)
            print(tagged)
            chunkGrams = '''Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'''
            chunkParser = nltk.RegexpParser(chunkGrams)

            chunked = chunkParser.parse(tagged)

            print(chunked)
    except Exception as e:
        print(str(e))

processContent()