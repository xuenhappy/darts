# 演示数据格式

本目录只演示词典与 Bigram 输入格式，不是默认模型的训练语料。文件均使用 UTF-8。

## 词典

`dregex_pattern_file.txt` 每行格式为 `LABELS:WORD`，多个标签使用逗号分隔，尖括号可省略：

```text
<CJK>,<ORG>:北京大学
<CJK>,<NLP>:自然语言处理
```

编译为紧凑 `.pbs`：

```bash
python scripts/devel.py dict-compile data/demo/dregex_pattern_file.txt /tmp/demo.pbs
python scripts/devel.py dict-benchmark /tmp/demo.pbs --text 北京大学自然语言处理
```

## Bigram

- `bigram_persenter_freq.txt`：制表符分隔的单词频率。
- `bigram_persenter_freqR.txt`：制表符分隔的相邻词联合频率。

生产资源由 `scripts/data_pipeline.py` 基于 `data/sources.json` 中固定版本的开放数据生成，来源、许可证、训练/开发/测试隔离规则见 [`../readme.md`](../readme.md)。
