import nltk, random

def readIMTCI():
    filename = 'train/Indonesian_Manually_Tagged_Corpus_ID_modified.tsv'
    with open(filename) as f:
        content = f.readlines()
        ctn = [nltk.tag.str2tuple(x.strip(), sep='\t') for x in content]
        training_sentences = []
        cur_sentence = []
        for tup in ctn:
            if tup[0][0:2] == '<k':
                cur_sentence = []
            elif tup[0][0:2] == '</':
                training_sentences.append(cur_sentence)
            else:
                cur_sentence.append(tup)
        
        return training_sentences

def readUGMCorpus():
    filename = 'train/ugm_postag_corpus.crp'
    with open(filename) as f:
        content = f.readlines()
        lines = [line.split(' ') for line in content]
        training_sentences = []
        for line in lines:
            training_sentences.append([nltk.tag.str2tuple(x, sep='/') for x in line])
        return training_sentences
        
def readPosTag(tagger):
    training_sentences = []
    training_sentences.extend(readIMTCI())
    training_sentences.extend(readUGMCorpus())

    if tagger == 'unigram':
        return nltk.UnigramTagger(training_sentences)
    elif tagger == 'bigram':
        return nltk.BigramTagger(training_sentences)
    else:
        return nltk.TrigramTagger(training_sentences)
