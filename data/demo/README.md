# 演示数据格式

本目录只演示词典与 Bigram 输入格式，不是默认模型的训练语料。文件均使用 UTF-8。

## 神经训练样本

`cws-train.txt` 和 `cws-dev.txt` 是小型格式示例，可用于快速检查 reader、前反向传播和 ONNX 导出，不能替代开放训练集。每行是一句话，金标词之间用一个或多个空白分隔；训练时会删除分隔空白再构造 Atom/WordPiece 对齐。

```text
南京市 长江 大桥 今日 正式 通车
目标 检测 模型 量化 可以 提高 推理 速度
```

- span 识别任务枚举 2 到 `max.span` 个 Atom 的所有区间；与任一金标词边界完全相同的区间标为 1。区间彼此独立，允许“南京”“南京市”“市长”等候选重叠，不生成 BIO。
- 图量化任务把词典候选、上述高召回 span、单 Atom 兜底和金标长词合并成 DAG。金标边标为 1，`GraphLossSparse` 使用边的关联概率负对数训练完整路径。
- train 用于参数更新，dev 只用于早停和按词长选择概率阈值。即使是 demo 也不要用 dev 反向更新，更不能用 test 调参。

快速 smoke test：

```bash
python scripts/train_joint.py train \
  --train data/demo/cws-train.txt --dev data/demo/cws-dev.txt \
  --epochs 1 --hidden-size 32 --recognizer-batch-size 4 \
  --quantizer-batch-size 2 --output-dir /tmp/darts-joint-demo
python scripts/train_joint.py export /tmp/darts-joint-demo/best.pt /tmp/darts-neural-demo
```

样本特意包含切分歧义、多音词、中英数字、2~5 字常见词和超过 5 字的长词，用于暴露边界/回退问题，不用于报告准确率。

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
