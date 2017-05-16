import nltk, pickle, sys
from nltk import word_tokenize
from readTraining import readPosTag, processTrainingData
from nltk.tokenize.moses import MosesDetokenizer

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

def tagTestSet(tagger):
    testingFile = "test/testing_data_new.txt"
    taggedTest = []
    with open(testingFile) as f:
        for idx, line in enumerate(f):
            taggedTest.append(joinTogether(tagger.tag(word_tokenize(line))))

    return taggedTest

def main():
    ptd = loadProcessedTrainingData()
    nerTagger = setupNerTagger(ptd)
    taggedTest = tagTestSet(nerTagger)
    
    for line in taggedTest:
        pass
        #print(line)

    jn = nerTagger.tag(word_tokenize("Budi, pergi budi BUDI dan bUdI sama BabaDi."))

    print(joinTogether(jn))

if __name__ == "__main__":
    main()