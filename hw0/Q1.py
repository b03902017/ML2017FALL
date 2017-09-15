import sys
words = {}
words_order = []
with open(sys.argv[1]) as f:
    for word in f.read().split():
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
            words_order.append(word)
with open('Q1.txt', 'w') as f:
    for index, word in enumerate(words_order):
        if index < len(words_order)-1:
            f.write('{word} {index} {fre}\n'\
                .format(word=word,index=index,fre=words[word]))
        else:
            f.write('{word} {index} {fre}'\
                .format(word=word,index=index,fre=words[word]))
