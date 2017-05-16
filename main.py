import nltk, pickle, sys
from nltk import word_tokenize
from readTraining import readPosTag, processTrainingData
from nltk.tokenize.moses import MosesDetokenizer
from crfsuite import *
from nltk.tag import CRFTagger

def loadProcessedTrainingData():
    trainingFiles = ["train/training_data_new.txt", "train/ugm_data_train.txt"]

    try:
        if len(sys.argv) > 1 and str.lower(sys.argv[1]) == 'reload':
            raise IOError()
        posTagger = pickle.load(open("pickles/posTagger.pickle", "rb"))
        ptd = pickle.load(open("pickles/ptd.pickle", "rb"))
        print("Loaded processed training data from dump!")
    except (OSError, IOError, ) as e:
        posTagger = readPosTag('unigram')
        ptd = processTrainingData(posTagger, trainingFiles)
        pickle.dump(ptd, open("pickles/posTagger.pickle", "wb"))
        pickle.dump(ptd, open("pickles/ptd.pickle", "wb"))
        print("Reloaded processed training data from dump!")

    return ptd

def joinTogether(words):
    justWords = [tup[0] if tup[1] is None else ("<ENAMEX TYPE=" + tup[1] + ">" + tup[0] + "</ENAMEX>") for tup in words]
    detokenizer = MosesDetokenizer()
    return detokenizer.detokenize(justWords, return_str=True)

def setupNerTagger(ptd):
    nerTagger = nltk.UnigramTagger(ptd.nerTaggedLines)
    return nerTagger

def joinTogetherWithTag(tokens, nerTags):
    n = len(tokens)
    finalWords = []
    inChunkFlag = False
    for i in range(n):
        if nerTags[i][0] == 'B':
            if inChunkFlag is not True:
                nerTag = nerTags[i].split("-")[1]
                finalWords.append("<ENAMEX TYPE=" + nerTag + ">" + tokens[i])
                inChunkFlag = True
            else:
                finalWords[-1] = finalWords[-1]+"</ENAMEX>"
        elif (nerTags[i][0] != 'I' and inChunkFlag == True):
            finalWords[-1] = finalWords[-1]+"</ENAMEX>"
            finalWords.append(tokens[i])
        elif i == n-1 and inChunkFlag == True:
            finalWords[i-1] = finalWords[i]+"</ENAMEX>"
        else:
            finalWords.append(tokens[i])
    return finalWords

def tagTestSet(tagger):
    testingFile = "test/testing_data_new.txt"
    taggedTest = []
    with open(testingFile) as f:
        for idx, line in enumerate(f):
            tokenizedLine = word_tokenize(line)
            nerTags = tagger.tag(tokenizedLine)
            # print("----------------------------")
            # print(tokenizedLine)
            # print(nerTags)
            taggedLine = joinTogetherWithTag(tokenizedLine, nerTags)

            detokenizer = MosesDetokenizer()
            taggedTest.append(detokenizer.detokenize(taggedLine, return_str=True))
            # taggedTest.append(joinTogether(tagger.tag(word_tokenize(line))))
    return taggedTest

def main():
    ptd = loadProcessedTrainingData()
    crfTagger = crf(ptd.iobTaggedLines)
    taggedTest = tagTestSet(crfTagger)

    for line in taggedTest:
        print(line)

    # ct = CRFTagger()
    # train_data = ptd.nerTaggedLines
    # ct.train(train_data, 'conll2002-esp.crfsuite')
    # print(ct.tag_sents(['Budi', 'membeli', 'Jokowi']))

    # ss = "Finalis Miss Indonesia Aktif Berdiskusi dengan Pemprov DKI: Agenda karantina Miss Indonesia 2015 diisi audie...  ?? ???? ?? B-)"
    # example_sent = ptd.iobTaggedLines[1]
    # # print(sent2features(example_sent))
    # posTagger = nltk.UnigramTagger(ptd.posTaggedLines)
    # # print(ptd.posTaggedLines[4])
    # example_sent = posTagger.tag(word_tokenize(ss))
    # sf = nltk.tag.crf.
    # print(example_sent)
    # example_sent = [(t[0],t[1],'O') if t[1] is not None else (t[0],'U','O') for t in example_sent]
    # print(example_sent)
    # print(' '.join(sent2tokens(example_sent)), end='\n\n')
    #
    # print("Predicted:", ' '.join(crfTagger.tag(sent2features(example_sent))))
    # print("Correct:  ", ' '.join(sent2labels(example_sent)))

    # for line in taggedTest:
    #     pass
    #     # print(line)

    # jn = nerTagger.tag(word_tokenize("Budi, pergi budi BUDI dan bUdI sama BabaDi."))
    #
    # print(joinTogether(jn))

if __name__ == "__main__":
    main()
