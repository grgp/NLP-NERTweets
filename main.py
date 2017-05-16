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

def setupNerTagger():    
    ptd = loadProcessedTrainingData()
    nerTagger = nltk.UnigramTagger(ptd.nerTaggedLines)
    return nerTagger

def joinTogether(words):
    justWords = [tup[0] if tup[1] is None or len(tup[1]) < 3 else ("<ENAMEX TYPE=" + tup[1] + ">" + tup[0] + "</ENAMEX>") for tup in words]
    detokenizer = MosesDetokenizer()
    return detokenizer.detokenize(justWords, return_str=True)

def main():
    nerTagger = setupNerTagger()
    jn = nerTagger.tag(word_tokenize("Budi, pergi ke pasar bersama Jokowi ke pasar Juventus bersama Andre orang Malaysia."))
    print(jn)
    print(joinTogether(jn))

if __name__ == "__main__":
    main()