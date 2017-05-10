import nltk, random

def trainPosTag(tagger):
    filename = 'Indonesian_Manually_Tagged_Corpus_ID_modified.tsv' # delete line 2833
    with open(filename) as f:
        content = f.readlines()

    ctn = [nltk.tag.str2tuple(x.strip(), sep='\t') for x in content]
    training_sentences = []
    testing_ids = random.sample(range(1, 10001), 0)
    testing_sentences = []
    testing_sentences_notag = []
    sentence_id = 0

    test_flag = False
    cur_sentence = []
    cur_sentence_notag = []
    for tup in ctn:
        if tup[0][0:2] == '<k':
            sentence_id = int(tup[0][0:-1].split(sep='<kalimat id=')[1])
            cur_sentence = []
            cur_sentence_notag = []
            if sentence_id in testing_ids:
                test_flag = True
        elif tup[0][0:2] == '</':
            if test_flag:
                testing_sentences.append(cur_sentence)
                testing_sentences_notag.append(cur_sentence_notag)
            else:
                training_sentences.append(cur_sentence)
            test_flag = False
        elif test_flag:
            cur_sentence.append(tup)
            cur_sentence_notag.append(tup[0])
        else:
            cur_sentence.append(tup)

    # ---- end of core program

    if tagger == 'unigram':
        return nltk.UnigramTagger(training_sentences)
    elif tagger == 'bigram':
        return nltk.BigramTagger(training_sentences)
    else:
        return nltk.TrigramTagger(training_sentences)