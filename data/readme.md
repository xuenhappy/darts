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
| `kernel/` | 字符类型、归一化和拼音等运行时核心数据 |
| `licenses/` | 随运行时衍生数据分发的上游许可证 |
| `codes/` | 模型编码资源 |
| `demo/` | 格式与编译命令演示，不作为默认模型训练集 |

CC BY-SA 数据的再分发和衍生使用必须保留署名及相同方式共享；具体版权文本由流水线下载到对应来源目录。

`kernel/pinyin.txt` 提供单字多音读法；`kernel/pinyin-phrases.txt` 是短语库与最终核心词典的交集，当前约 2.2 万条。拼音模式优先使用词级读音消除歧义，未命中时逐字回退。非 CJK 默认不添加拼音；如业务需要统一占位符，可在 `pinyin.dict` 中配置 `"non-cjk.label": "<NO_PINYIN>"`。
