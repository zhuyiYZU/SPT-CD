from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format('./data/crawl-300d-2M-subword/crawl-300d-2M-subword.vec', binary=False)

# print(model.similar_by_word('news', 50))

l_word = []
res1 = []
res2 = []
res3 = []
with open('verbalizer/cpt_verbalizer2.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        l_word.append(line.split(',')[0].strip())
        res1.append(line.split(',')[1:])

with open('result/bert_result.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        res2.append(line.split(',')[1:])

with open('result/fasttext_result.txt', 'w') as f:

    for i, word in enumerate(l_word):
        l_temp = []
        for word1, _ in model.similar_by_word(word, 500):
            l_temp.append(word1.lower())
        f.write(','.join(l_temp) + '\n')

        res3.append(set(l_temp) & set(res1[i]) & set(res2[i]))

with open('result/all_result.txt', 'w') as f:
    for i, word in enumerate(l_word):
        f.write(word + ',' + ','.join(list(res3[i])) + '\n')

