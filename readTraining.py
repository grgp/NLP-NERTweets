import nltk, random
from nltk.tokenize import word_tokenize
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
    with open("training_data_new.txt") as f:
        errorCount = 0
        for idx, line in enumerate(f):
            try:
                if idx == 452:
                    taggedWords = toNERTaggedTuples(line)
                    justWords = [tuple[0] for tuple in taggedWords]

                    j = joinBackTogether(justWords)
                    k = trainPosTag('unigram').tag(justWords)
                    print(str(len(taggedWords)) + ' tokens: ' + str(taggedWords))
                    print(str(len(k)) + ' tokens: ' + str(k))
            except UnicodeEncodeError:
                print("Error yo, I'm counting dis.")
                errorCount += 1

        print("errorCount: " + str(errorCount))

if __name__ == "__main__":
    main()