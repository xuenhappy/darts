from .darts.devel import *
from .cdarts import *
from .darts import wtype


def charSplit(unicode_strs):
    """
    this only split the string by the chartype and noly return one type\n
    if you modify this code,must avoid return to many type\n
    other type better add in segment cellregnizer
    """
    chr_buffer = []
    buf_type = None
    for ch in unicode_strs:
        ctype = wordType(ch)
        if ctype != buf_type:
            if (ctype == wtype.EMPTY and buf_type == wtype.POS):
                buf_type = wtype.POS
                chr_buffer.append(ch)
                continue
            if (buf_type == wtype.EMPTY and ctype == wtype.POS):
                buf_type = wtype.POS
                chr_buffer.append(ch)
                continue

            if chr_buffer:
                yield "".join(chr_buffer), buf_type
            chr_buffer = []

        buf_type = ctype

        if buf_type == wtype.CJK:
            yield ch, buf_type
            continue
        chr_buffer.append(ch)

    if chr_buffer:
        yield "".join(chr_buffer), buf_type


def _static_words(cfile):
    words_freq = {}
    with open(cfile) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            for w, wt in charSplit(line):
                if wt != wtype.ENG:
                    continue
                words_freq[w] = words_freq.get(w, 0)+1
    return words_freq


def bpe_train(text_corpus_path, mini_freq, max_words_num, topk_percent, min_pmi, min_lw):
    """对文本预料进行新词发现，并在当前目录输出词频文件与二元语法统计词表
    算法参见 https://zhuanlan.zhihu.com/p/80385615
    """
    print(f"static the word freq from file {text_corpus_path}")
    words_freq = _static_words(text_corpus_path)
    print(f"filter the mini freq {mini_freq} words..")
    words_freq = {k: v for k, v in words_freq.items() if v > mini_freq}
    
