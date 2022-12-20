#! -*- coding: utf-8 -*-

import codecs
import math
import sys
from ..cdarts import PyAtomList


def filter_ngrams(ngrams, total, min_pmi=1):
    """通过互信息过滤ngrams，只保留“结实”的ngram。
    """
    order = len(ngrams)
    if hasattr(min_pmi, '__iter__'):
        min_pmi = list(min_pmi)
    else:
        min_pmi = [min_pmi] * order
    output_ngrams = set()
    total = float(total)
    for i in range(order - 1, 0, -1):
        for w, v in ngrams[i].items():
            pmi = min([
                total * v / (ngrams[j].get(w[:j + 1], total) * ngrams[i - j - 1].get(w[j + 1:], total))
                for j in range(i)
            ])
            if math.log(pmi) >= min_pmi[i]:
                output_ngrams.add(w)
    return output_ngrams


class SimpleTrie:
    """通过Trie树结构，来搜索ngrams组成的连续片段
    """

    def __init__(self):
        self.dic = {}
        self.end = True

    def add_word(self, word):
        _ = self.dic
        for c in word:
            if c not in _:
                _[c] = {}
            _ = _[c]
        _[self.end] = word

    def tokenize(self, sent):
        result = []
        start, end = 0, 1
        for i, c1 in enumerate(sent):
            _ = self.dic
            if i == end:
                result.append(sent[start:end])
                start, end = i, i + 1
            for j, c2 in enumerate(sent[i:]):
                if c2 in _:
                    _ = _[c2]
                    if self.end in _:
                        if i + j + 1 > end:
                            end = i + j + 1
                else:
                    break
        result.append(sent[start:end])
        return result


def filter_vocab(candidates, ngrams, order):
    """通过与ngrams对比，排除可能出来的不牢固的词汇(回溯)
    """
    result = {}
    for i, j in candidates.items():
        if len(i) < 3:
            result[i] = j
        elif len(i) <= order and i in ngrams:
            result[i] = j
        elif len(i) > order:
            flag = True
            for k in range(len(i) + 1 - order):
                if i[k:k + order] not in ngrams:
                    flag = False
            if flag:
                result[i] = j
    return result


min_count = 32
order = 4
corpus_file = 'thucnews.corpus'  # 语料保存的文件名
vocab_file = 'thucnews.chars'  # 字符集保存的文件名
ngram_file = 'thucnews.ngrams'  # ngram集保存的文件名
output_file = 'thucnews.vocab'  # 最后导出的词表文件名

write_corpus(text_generator(), corpus_file)  # 将语料转存为文本
count_ngrams(corpus_file, order, vocab_file, ngram_file, memory)  # 用Kenlm统计ngram
ngrams = KenlmNgrams(vocab_file, ngram_file, order, min_count)  # 加载ngram
ngrams = filter_ngrams(ngrams.ngrams, ngrams.total, [0, 2, 4, 6])  # 过滤ngram
ngtrie = SimpleTrie()  # 构建ngram的Trie树

for w in Progress(ngrams, 100000, desc=u'build ngram trie'):
    _ = ngtrie.add_word(w)

candidates = {}  # 得到候选词
for t in Progress(text_generator(), 1000, desc='discovering words'):
    for w in ngtrie.tokenize(t):  # 预分词
        candidates[w] = candidates.get(w, 0) + 1

# 频数过滤
candidates = {i: j for i, j in candidates.items() if j >= min_count}
# 互信息过滤(回溯)
candidates = filter_vocab(candidates, ngrams, order)

# 输出结果文件
with codecs.open(output_file, 'w', encoding='utf-8') as f:
    for i, j in sorted(candidates.items(), key=lambda s: -s[1]):
        s = '%s %s\n' % (i, j)
        f.write(s)


def static_word_freq(wordfile):
    wdict = {}
    with codecs.open(wordfile, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            for atom in PyAtomList(line.lower()).tolist():
                if 'ENG' != atom.chtype:
                    continue
                wdict[atom.image] = wdict.get(atom.image, 0) + 1
    return wdict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s [text corpus]" % sys.argv[0])
        exit(0)
    print("start static word freq...")
    wdict = static_word_freq(sys.argv[1])
    
    print("fileter mini freq...")
    print("static wordpice ngram....")
    print("save ngram data...")
