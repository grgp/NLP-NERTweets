with open("training_data_new.txt") as f:
    errorCount = 0
    for idx, line in enumerate(f):
        try:
            if idx == 456:
                print(line)
                taggedWords = []
                taggedRaw = line.split('<ENAMEX TYPE="')
                if len(taggedRaw[0]) > 0:
                    residue = taggedRaw[0].strip().split(' ')
                    taggedWords.extend([(word, 'X') for word in residue])
                print(taggedWords)
                for fragment in taggedRaw[1:]:
                    residue = fragment.split('</ENAMEX>')
                    th = residue[0].strip().split('">')
                    taggedWords.append((th[1], th[0]))
                    
                    filtered = [(word, 'X') for word in residue[1].split(' ') if len(word) > 0]
                    taggedWords.extend(filtered)
                print(taggedWords)
        except UnicodeEncodeError:
            print("Error yo")
            errorCount += 1

    print("errorCount: " + str(errorCount))

# s = 'Batu <E="Org">FBI</E> adalah <E="Prs">Bro</E> kapas.'
# print(str(s.split('<E="')))