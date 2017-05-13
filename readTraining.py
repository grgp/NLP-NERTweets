import nltk, pickle, sys
from nltk import word_tokenize, pos_tag, ne_chunk
from readPosTag import readPosTag

def postgd(line):
    return word_tokenize(line.strip())

def joinBackTogether(words):
    s = ""
    for word in words:
        if len(word) == 1 and word[0] in [',', '.', '!', '?', ':', ';']:
            s += word
        else:
            s += ' ' + word
    return s.strip()

class processedTrainingData:
    nerTaggedLines = []
    posTaggedLines = []
    nerWordset = {}
    posWordset = {}

    def __init__(self, nerTaggedLines=[], posTaggedLines=[], nerWordset={}, posWordset={}):
        self.nerTaggedLines = nerTaggedLines
        self.posTaggedLines = posTaggedLines
        self.nerWordset = nerWordset
        self.posWordset = posWordset    

def trainingDataToNERTaggedTuples(line):
    taggedWords = []
    taggedRaw = line.split('<ENAMEX TYPE="')

    # add tokens left of first ENAMEX to taggedWords
    if len(taggedRaw[0]) > 0:
        residue = taggedRaw[0].strip().split(' ')
        taggedWords.extend([(word, 'X') for word in residue])

    # add ENAMEX tokens
    for fragment in taggedRaw[1:]:
        residue = fragment.split('</ENAMEX>')
        th = residue[0].strip().split('">')
        taggedWords.append((th[1], th[0]))
        
        # add tokens right of ENAMEX
        filtered = [(word, 'X') for word in residue[1].split(' ') if len(word) > 0]
        taggedWords.extend(filtered)

    return taggedWords

def processTrainingData(posTagger, trainingFiles):
    nerTaggedLines = []
    posTaggedLines = []
    nerWordset = {}
    posWordset = {}

    for trainingFile in trainingFiles:
        with open(trainingFile) as f:
            for idx, line in enumerate(f):
                try:
                    nerTaggedWords = trainingDataToNERTaggedTuples(line)

                    justWords = [tuple[0] for tuple in nerTaggedWords]
                    posTaggedWords = posTagger.tag(justWords)
                    
                    nerTaggedLines.append(nerTaggedWords)
                    posTaggedLines.append(posTaggedWords)

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
        
        return processedTrainingData(nerTaggedLines, posTaggedLines, nerWordset, posWordset)