import nltk
from nltk.tokenize import word_tokenize

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
    if len(taggedRaw[0]) > 0:
        residue = taggedRaw[0].strip().split(' ')
        taggedWords.extend([(word, 'X') for word in residue])
    for fragment in taggedRaw[1:]:
        residue = fragment.split('</ENAMEX>')
        th = residue[0].strip().split('">')
        taggedWords.append((th[1], th[0]))
        
        filtered = [(word, 'X') for word in residue[1].split(' ') if len(word) > 0]
        taggedWords.extend(filtered)

    return taggedWords

with open("training_data_new.txt") as f:
    errorCount = 0
    for idx, line in enumerate(f):
        try:
            if idx == 456:
                taggedWords = toNERTaggedTuples(line)
                justWords = [tuple[0] for tuple in taggedWords]

                print(taggedWords)
                print(joinBackTogether(justWords))
        except UnicodeEncodeError:
            print("Error yo, I'm counting dis.")
            errorCount += 1

    print("errorCount: " + str(errorCount))