# 演示数据

本目录提供可直接用于词典编译、规则验证、神经训练和 ONNX 导出的最小数据集。所有
文件使用 UTF-8，仅用于格式、位置对齐和训练流程 smoke test，不用于报告准确率。

## 文件索引

| 文件 | 用途 | 主要消费者 |
| --- | --- | --- |
| `dregex_pattern_file.txt` | 通用语义类型词典 | `dict-compile`、`DictWordRecongnizer` |
| `lac-dictionary.txt` | LAC/POS 词典 | `DictWordRecongnizer(atom.mode=true)` |
| `cws-train.txt` / `cws-dev.txt` | 成词二分类与通用图量化训练 | `SpanSampleReader`、`GraphSampleReader` |
| `lac-train.txt` / `lac-dev.txt` | `NOT_WORD + POS` 多分类与 LAC 图量化训练 | `SyntaxSpanSampleReader`、`GraphSampleReader` |
| `bigram_persenter_freq.txt` | 单词频率 | `build_gramdict_fromfile` |
| `bigram_persenter_freqR.txt` | 相邻词频率 | `build_gramdict_fromfile` |
| `pinyin-phrases.txt` | 词级拼音消歧 | `PinyinEncoder` 数据格式参考 |
| `location-dictionary.txt` | 省市区街道 POI 角色词典 | 地址模式数据格式参考 |
| `temporal-quantity.txt` | 中英文日期、时间、数量、单位文本 | `TemporalQuantityRecongnizer` |

训练集与开发集句子不重复。多个文件围绕同一组词语构造，方便检查词典召回、训练金标
和量化器候选图是否一致。

## 通用词典

`dregex_pattern_file.txt` 每行格式为：

```text
<LABEL>,<LABEL>:WORD
```

标签来自 `data/codes/type.hx.txt` 所使用的通用语义类型。注释行和空行会被开发脚本
跳过。

```bash
python3 scripts/devel.py dict-compile \
  data/demo/dregex_pattern_file.txt /tmp/demo-general.pbs
python3 scripts/devel.py dict-benchmark \
  /tmp/demo-general.pbs --text 北京大学发布中文分词模型
```

## LAC/POS 词典

`lac-dictionary.txt` 使用 `POS_*` 标签，与 `data/codes/pos.hx.txt`、
`pos.encoder` 和多分类模型标签一致。运行时应配置：

```jsonc
{
  "type": "DictWordRecongnizer",
  "pbfile.path": "/tmp/demo-lac.pbs",
  "atom.mode": "true"
}
```

`atom.mode=true` 会为同一 span 的每个 POS 标签保留独立候选，量化器节点使用
`(atom_start, atom_end, pos_type)` 区分这些候选。

```bash
python3 scripts/devel.py dict-compile \
  data/demo/lac-dictionary.txt /tmp/demo-lac.pbs
```

## CWS 二分类语料

`cws-train.txt` 和 `cws-dev.txt` 每行是一句话，词语由一个或多个空白分隔：

```text
北京大学 发布 中文分词 模型
```

`SpanSampleReader` 枚举候选 span，将金标词标为 `1`、其他区间标为 `0`。候选彼此
独立，不使用 BIO 或 CRF。`GraphSampleReader` 使用同一句子的金标边训练量化器，
量化器输出 `-log P(next | previous)`。

```bash
PYTHON=/home/xuen/.venv/bin/python python3 scripts/devel.py joint-train \
  --recognizer-kind binary \
  --train data/demo/cws-train.txt \
  --dev data/demo/cws-dev.txt \
  --epochs 1 --hidden-size 32 \
  --output-dir /tmp/darts-binary-joint
```

## LAC 多分类语料

`lac-train.txt` 和 `lac-dev.txt` 使用空白分隔的 `词/POS_*` 格式：

```text
北京大学/POS_PROPN 发布/POS_VERB 中文分词/POS_NOUN 模型/POS_NOUN
```

`SyntaxSpanSampleReader` 的类别 `0` 固定为 `NOT_WORD`，其他类别来自
`data/codes/pos.hx.txt`。识别头和图量化器共享同一 `WordEncoder`；图量化器使用
`pos.encoder` 对候选词性编码。

```bash
PYTHON=/home/xuen/.venv/bin/python python3 scripts/devel.py joint-train \
  --recognizer-kind syntax \
  --train data/demo/lac-train.txt \
  --dev data/demo/lac-dev.txt \
  --epochs 1 --hidden-size 32 \
  --output-dir /tmp/darts-lac-joint

PYTHON=/home/xuen/.venv/bin/python python3 scripts/devel.py joint-export \
  /tmp/darts-lac-joint/best.pt /tmp/darts-lac-models
```

多分类导出 `syntax.onnx`、`syntax.labels.txt`、`lac-indicator.onnx` 和
`lac-quantizer.onnx`，对应默认 `lac-nural` 模式。

## Atom 与空白对齐

标准语料中的空白只表示词边界，不进入 AtomList。reader 使用：

```text
skip_space=true
normal_before=false
```

整句先原子化，再将金标字符边界映射到 Atom 边界。相邻英文或数字词之间的语料空白
会阻止它们被合并成一个 Atom，但 `best_path`、候选节点和金标节点索引中均不包含该
空白。WordPiece span 使用首尾闭区间，Atom span 使用首闭尾开区间。

## Bigram

- `bigram_persenter_freq.txt`：`WORD<TAB>FREQUENCY`，可附加第三列标签。
- `bigram_persenter_freqR.txt`：`LEFT-RIGHT<TAB>FREQUENCY`。

示例频率与 CWS/LAC 句子一致，可验证编译和路径代价，但不是实际语料统计值。

```bash
python3 scripts/devel.py bigram-compile \
  data/demo/bigram_persenter_freq.txt \
  data/demo/bigram_persenter_freqR.txt \
  /tmp/demo.bdf
```

## 拼音、地址与规则

`pinyin-phrases.txt` 使用 `词语: 带声调数字的音节序列`，用于展示“重庆、银行、
行长、音乐”等多音词的句子级消歧格式。非中文词不要求拼音。

`location-dictionary.txt` 展示省、市、区、街道和 POI 标签。地址模式还必须启用
`AddressRecongnizer` 规则，不能只依赖词典。

`temporal-quantity.txt` 包含中英文日期时间、数量、货币/存储/公制单位的规则输入。
它是原始句子集合，不是空白分词训练语料。

生产资源由 `scripts/data_pipeline.py` 根据 `data/sources.json` 中固定版本的开放
数据生成。来源、许可证及 train/dev/test 隔离规则见 [`../readme.md`](../readme.md)。
