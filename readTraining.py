# with open("training_data_new.txt") as f:
#     errorCount = 0
#     for idx, line in enumerate(f):
#         try:
#             line.split('<ENAMEX TYPE="')[1].split('</ENAMEX>')[0].split('">')
#         except UnicodeEncodeError:
#             print("Error yo")
#             errorCount += 1

#     print("errorCount: " + str(errorCount))

s = 'Batu <E="Org">FBI</E> adalah <E="Prs">Bro</E> kapas.'
print(str(s.split('<E="')))