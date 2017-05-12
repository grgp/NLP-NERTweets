import nltk, random, timeit
from nltk import word_tokenize, pos_tag, ne_chunk
from wsPostag import trainPosTag

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

def toNERTaggedTuples(line):
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

def main():
    tagger = trainPosTag('unigram')
    wordset = {}
    with open("training_data_new.txt") as f:
        errorCount = 0
        taggedWordsCount = 0
        untaggedWordsCount = 0
        taggedNNPCount = 0
        taggedAsNNP = 0
        notTaggedAsNNP = 0
        for idx, line in enumerate(f):
            try:
                if True:
                    taggedWords = toNERTaggedTuples(line)
                    justWords = [tuple[0] for tuple in taggedWords]

                    j = joinBackTogether(justWords)
                    k = tagger.tag(justWords)
                    
                    for idx, tup in enumerate(taggedWords):
                        if tup[1] != 'X':
                            if k[idx][1] == 'NNP':
                                taggedAsNNP += 1
                            else:
                                notTaggedAsNNP += 1
                            lw = str.lower(tup[0])
                            if lw in wordset:
                                wordset[lw] += 1
                            else:
                                wordset[lw] = 1

                    for tup in k:
                        if tup[1] is not None:
                            taggedWordsCount += 1
                            if tup[1] == 'NNP':
                                taggedNNPCount += 1
                        else:
                            untaggedWordsCount += 1

            except UnicodeEncodeError:
                print("Error yo, I'm counting dis.")
                errorCount += 1

        print("unicodeErrorCount: " + str(errorCount))
        print()
        print("taggedWordsCount: " + str(taggedWordsCount))
        print("untaggedWordsCount: " + str(untaggedWordsCount))
        print("percentageCounted: " + str(taggedWordsCount/(taggedWordsCount+untaggedWordsCount)))
        print()
        print("taggedNNPCount: " + str(taggedNNPCount))
        print("percentageNNPCounted: " + str(taggedAsNNP/(taggedAsNNP+notTaggedAsNNP)))
        print()
        print(wordset)

if __name__ == "__main__":
    main()