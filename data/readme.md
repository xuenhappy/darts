# 数据与模型

默认分词资源由固定版本的开放数据生成，不再依赖 `data/demo` 中规模有限的演示样本。

## 数据来源

| 来源 | 版本 | 许可 | 用途 |
| --- | --- | --- | --- |
| UD Chinese GSD | UD 2.18 | CC BY-SA 4.0 | 训练、开发、测试切分与 Bigram 统计 |
| jieba `dict.txt` | v0.42.1 | MIT | 扩展通用中文词典覆盖率 |
| phrase-pinyin-data | v0.19.0 | MIT | 词级拼音与多音字消歧 |

下载地址记录在 `sources.json`，实际文件大小与 SHA-256 记录在 `sources.lock.json`。原始数据和生成中间文件分别位于被 Git 忽略的 `external/`、`generated/`；仓库只提交生成后的运行时模型。

## 重建

```bash
python scripts/data_pipeline.py download
python scripts/data_pipeline.py prepare
python scripts/data_pipeline.py build-models

# 等价的统一开发入口
python scripts/devel.py data download
python scripts/devel.py data prepare
python scripts/devel.py data build-models
```

默认构建无条件保留 UD train 词汇，只接收 jieba 中频率至少为 10 的外部词，并过滤超过 8 个 Unicode 字符的异常长词。参数可通过 `prepare --jieba-min-frequency` 和 `--max-word-length` 调整；只能用 dev 选择参数，不能反复查看 test 调参。

下载遵循 `HTTP_PROXY`、`HTTPS_PROXY` 和小写同名环境变量，并支持 `.part` 断点续传。网络较慢时可先配置代理。生成后将 `generated/models/open-dictionary.pbs` 和 `open-bigram.bdf` 更新到 `models/`，再执行完整测试。

## 评测

```bash
python scripts/data_pipeline.py evaluate data/generated/cws-dev.txt --mode faster
python scripts/data_pipeline.py evaluate data/generated/cws-test.txt --mode faster
```

评测按 Atom 边界计算 precision、recall、F1 和整句完全匹配率。开发集只能用于选择配置，最终结果应以未参与调参的 test 切分为准。

## 目录

| 路径 | 说明 |
| --- | --- |
| `conf.json` | 默认运行配置，`hybrid` 为默认模式 |
| `sources.json` | 固定版本的上游数据清单 |
| `sources.lock.json` | 已下载文件的大小与 SHA-256 |
| `models/panda.pbs` | 开放词典编译后的紧凑 Trie |
| `models/ngram_dict.bdf` | UD 训练集生成的 Bigram 数据 |
| `models/neural/recognizer.onnx` | 2~5 Atom 可重叠候选的成词概率模型 |
| `models/neural/indicator.onnx` | 量化器独立训练的 Transformer 词表示网络 |
| `models/neural/quantizer.onnx` | 相邻词关联概率负对数网络 |
| `kernel/` | 字符类型、归一化和拼音等运行时核心数据 |
| `licenses/` | 随运行时衍生数据分发的上游许可证 |
| `codes/` | 模型编码资源 |
| `demo/` | 神经训练格式与词典编译命令演示，不作为默认生产训练集 |

CC BY-SA 数据的再分发和衍生使用必须保留署名及相同方式共享；具体版权文本由流水线下载到对应来源目录。

`kernel/pinyin.txt` 提供单字多音读法；`kernel/pinyin-phrases.txt` 是短语库与最终核心词典的交集，当前约 2.2 万条。拼音模式优先使用词级读音消除歧义，未命中时逐字回退。非 CJK 默认不添加拼音；如业务需要统一占位符，可在 `pinyin.dict` 中配置 `"non-cjk.label": "<NO_PINYIN>"`。

## 神经训练数据

识别器从 CWS 行格式生成全部 2~5 Atom span，独立标注每个 span 是否为词，不生成 BIO 序列。量化器训练图合并词典候选、同一 span 上界和金标长词，确保 OOV 金标路径始终存在；每条边的模型输出是 `-log P(next|previous)`，由 `GraphLossSparse` 在完整 DAG 上归一化。推荐使用联合训练，让两个任务损失同时更新一份 `WordEncoder` 参数；两个任务头独立，导出仍生成三个独立 ONNX。独立训练入口保留用于对照实验。

当前 train 切分包含 98,614 个词，其中 1~5 字词约占 99.5%；因此默认 `max.span=5`。超过 5 字的少量金标不会作为神经 recognizer 的穷举候选，但仍保留在核心词典和量化器金标图中。若业务长实体较多，应在业务 dev 集提高 `max.span` 并重新校准各长度阈值，而不是直接在 test 集调参。

```bash
python scripts/train_recognizer.py train --device cuda
python scripts/train_recognizer.py export model_bin/recognizer/best.pt data/models/neural/recognizer.onnx
python scripts/train_quantizer.py train --device cuda
python scripts/train_quantizer.py export model_bin/quantizer/best.pt data/models/neural

# 推荐的共享参数多任务训练
python scripts/train_joint.py train --device cuda
python scripts/train_joint.py export model_bin/joint/best.pt data/models/neural
```
