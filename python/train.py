from .darts.devel import *
from .cdarts import *


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
            if (ctype == '<EMPTY>' and buf_type == '<POS>'):
                buf_type = '<POS>'
                chr_buffer.append(ch)
                continue
            if (buf_type == '<EMPTY>' and ctype == '<POS>'):
                buf_type = '<POS>'
                chr_buffer.append(ch)
                continue

            if chr_buffer:
                yield "".join(chr_buffer), buf_type
            chr_buffer = []

        buf_type = ctype

        if buf_type == "<CJK>":
            yield ch, buf_type
            continue
        chr_buffer.append(ch)

    if chr_buffer:
        yield "".join(chr_buffer), buf_type


def bpe_train(text_corpus_path, mini_freq, max_words_num, topk_percent, min_pmi, min_lw):
    """对文本预料进行新词发现，并在当前目录输出词频文件与二元语法统计词表
    算法参见 https://zhuanlan.zhihu.com/p/80385615
    """
    pass
