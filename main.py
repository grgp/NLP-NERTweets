import nltk, pickle, sys
from nltk import word_tokenize
from readTraining import readPosTag, processTrainingData

def main():    
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
    
    nerTagger = nltk.UnigramTagger(ptd.nerTaggedLines)
    
    jn = nerTagger.tag(word_tokenize("Budi pergi ke pasar bersama Jokowi ke pasar Juventus bersama Andre orang Malaysia."))
    print(jn)

if __name__ == "__main__":
    main()