import nltk, pickle, sys
from nltk import word_tokenize, pos_tag, ne_chunk
from readPostag import readPosTag

class processedTrainingData:
    trainingLines = []
    nerTaggedLines = []
    posTaggedLines = []
    nerWordset = {}
    posWordset = {}
    iobTaggedLines = []

    def __init__(self, trainingLines=[], nerTaggedLines=[], posTaggedLines=[], nerWordset={}, posWordset={}, iobTaggedLines=[]):
        self.trainingLines = trainingLines
        self.nerTaggedLines = nerTaggedLines
        self.posTaggedLines = posTaggedLines
        self.nerWordset = nerWordset
        self.posWordset = posWordset
        self.iobTaggedLines = iobTaggedLines

def trainingDataToNERTaggedTuples(line):
    taggedWords = []
    taggedRaw = line.split('<ENAMEX TYPE="')

    # add tokens left of first ENAMEX to taggedWords
    if len(taggedRaw[0]) > 0:
        residue = word_tokenize(taggedRaw[0])
        taggedWords.extend([(word, None) for word in residue])

    # add ENAMEX tokens
    for fragment in taggedRaw[1:]:
        residue = fragment.split('</ENAMEX>')
        th = residue[0].strip().split('">')
        taggedWords.append((th[1], th[0]))

        # add tokens right of ENAMEX
        filtered = [(word, None) for word in word_tokenize(residue[1]) if len(word) > 0]
        taggedWords.extend(filtered)

    return taggedWords

def convertToIOB(nerTaggedWords, posTaggedWords):
    n = len(nerTaggedWords)
    iobTaggedWords = []
    for i in range(n):
        if nerTaggedWords[i][1] is not None:
            iob_tag = "B"
            chunks = nerTaggedWords[i][0].split(" ")
            if len(chunks) == 1:
                iobTaggedWords.append((nerTaggedWords[i][0], "NNP", "B-"+nerTaggedWords[i][1]))
            else:
                iobTaggedWords.append((chunks[0], "NNP", "B-"+nerTaggedWords[i][1]))
                for j in range(1, len(chunks)):
                    iobTaggedWords.append((chunks[j], "NNP", "I-"+nerTaggedWords[i][1]))
        else:
            if posTaggedWords[i][1] is not None:
                iobTaggedWords.append((nerTaggedWords[i][0], posTaggedWords[i][1], "O"))
            else:
                iobTaggedWords.append((nerTaggedWords[i][0], "U", "O"))

    return iobTaggedWords

def processTrainingData(posTagger, trainingFiles):
    trainingLines = []
    nerTaggedLines = []
    posTaggedLines = []
    nerWordset = {}
    posWordset = {}
    iobTaggedLines = []

    for trainingFile in trainingFiles:
        with open(trainingFile) as f:
            for idx, line in enumerate(f):
                trainingLines.append(line)
                try:
                    nerTaggedWords = trainingDataToNERTaggedTuples(line)
                    nerTaggedWords.append((".", None))
                    nerTaggedWords.extend([(tup[0].lower(), tup[1]) for tup in nerTaggedWords])

                    justWords = [tuple[0] for tuple in nerTaggedWords]
                    posTaggedWords = posTagger.tag(justWords)

                    iobTaggedWords = convertToIOB(nerTaggedWords, posTaggedWords)

                    nerTaggedLines.append(nerTaggedWords)
                    posTaggedLines.append(posTaggedWords)
                    iobTaggedLines.append(iobTaggedWords)

                    for idx, tup in enumerate(nerTaggedWords):
                        if tup[1] != 'X':
                            if tup in nerWordset:
                                nerWordset[tup] += 1
                            else:
                                nerWordset[tup] = 1

                    for idx, tup in enumerate(posTaggedWords):
                        if tup[1] == 'NNP':
                            if tup in posWordset:
                                posWordset[tup] += 1
                            else:
                                posWordset[tup] = 1

                except UnicodeEncodeError:
                    print("Error yo, I'm counting dis.")
                    errorCount += 1

        return processedTrainingData(trainingLines, nerTaggedLines, posTaggedLines, nerWordset, posWordset, iobTaggedLines)
