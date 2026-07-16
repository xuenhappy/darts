# Darts

Darts 是一个面向中文文本处理的 C++/Python 分词框架。它将字符归一化、原子切分、词典匹配、ONNX 序列识别、候选词图构建和最优路径搜索组合为一条可配置流水线，适合需要自定义词典、模型和切分策略的场景。

项目核心使用 C++17 实现，通过 C API 暴露稳定边界，并使用 Cython 生成 Python 扩展。默认发布物是包含模型、词典和必要动态库的 Linux wheel。

## 主要能力

- UTF-8 文本归一化和字符类型识别。
- 中文、英文、数字、空白和标点的原子级切分。
- 基于双数组 Trie/词典的候选词识别。
- 基于 ONNX 序列标注模型的新词候选识别。
- 最小覆盖、Bigram 和 ONNX 图打分决策器。
- 候选词图上的最短路径解码。
- 拼音编码、WordPiece 编码和词类型编码。
- 双数组模式匹配、词典编译和 Bigram 词典生成工具。
- C、C++ 和 Python 接口。
- PyTorch 训练、图损失与 ONNX 导出开发模块。

## 处理流程

```text
UTF-8 文本
    |
    v
归一化与原子切分 AtomList
    |
    v
基础单原子候选 + 多个 Recognizer 扩展候选
    |
    v
SegPath 候选词有向无环图
    |
    v
Decider 嵌入与边代价计算
    |
    v
Dijkstra 最短路径
    |
    v
WordList 分词结果
```

每个原子都会形成一个基础候选，因此即使词典或模型没有命中，候选图仍有基本路径。Recognizer 负责增加跨原子的词候选，Decider 负责计算相邻候选之间的非负代价，最后由最短路径选择总代价最低的完整切分。

## 目录结构

```text
.
├── data/                    运行时配置、词典、编码表和 ONNX 模型
├── src/
│   ├── core/                Atom、Word、SegPath、Segment 和图搜索
│   ├── impl/                配置加载、识别器、决策器、编码器和 ONNX 插件
│   ├── main/                C API 及核心实现入口
│   └── utils/               Trie、Bigram、拼音、正则、Proto 和文本工具
├── python/
│   ├── darts/               Cython API 与 Python 开发模块
│   ├── pyproject.toml        PEP 517 隔离构建依赖
│   └── setup.py             Meson 驱动的 wheel 打包逻辑
├── scripts/                 依赖引导、ONNX Runtime 下载和一键构建
├── test/                    C++ 功能测试
├── meson.build              原生构建入口
└── meson_options.txt        Meson 构建选项
```

## 环境要求

### 推荐环境

- Linux x86_64。
- Ubuntu/Debian 及兼容 `apt`、`dpkg-deb` 的发行版。
- GCC 或 Clang，支持 C++17。
- Python 3.8 及以上，包含 `venv` 和开发头文件。
- Meson 1.3 及以上。
- curl、pkg-config、protoc 和常用编译工具。
- 可访问 PyPI、Ubuntu/Debian 软件源和 GitHub Release。

自动依赖脚本目前下载 `onnxruntime-linux-x64`，因此 ARM64、macOS 和 Windows 不能直接使用该自动下载流程。其他平台需要自行准备 ONNX Runtime，并通过 `DARTS_ONNXRUNTIME_DIR` 指定安装前缀。

### 系统安装方式

Ubuntu/Debian 可以先安装系统依赖：

```bash
./scripts/bootstrap_linux_deps.sh
python3 -m pip install --user "meson>=1.3"
```

也可以手动安装：

```bash
sudo apt update
sudo apt install -y \
  build-essential pkg-config protobuf-compiler \
  libprotobuf-dev libjsoncpp-dev libfmt-dev zlib1g-dev \
  python3-dev python3-venv python3-pip curl ca-certificates

python3 -m pip install --user "meson>=1.3" "ninja>=1.11"
```

确认工具可用：

```bash
python3 --version
meson --version
pkg-config --version
protoc --version
```

## 一键编译、打包和测试

这是当前推荐且已经验证的构建方式。脚本会：

1. 下载 ONNX Runtime 1.17.3 到 `build/deps/`。
2. 使用 `apt download` 下载 C++ 依赖并展开到用户目录 sysroot，不修改系统软件包。
3. 创建隔离构建虚拟环境。
4. 使用 Meson 编译 C++ 静态库和 Cython 扩展。
5. 生成 wheel 到 `python/dist/`。
6. 在新的测试虚拟环境中安装 wheel 并执行导入测试。

```bash
git clone <repository-url> darts
cd darts
./scripts/build_all.sh --clean --test
```

常用参数：

```bash
# 增量构建并生成 wheel
./scripts/build_all.sh

# 清理历史构建后重新生成
./scripts/build_all.sh --clean

# 清理、构建、安装并验证
./scripts/build_all.sh --clean --test

# 查看帮助
./scripts/build_all.sh --help
```

默认产物：

```text
python/dist/darts-<version>-<python-tag>-linux_x86_64.whl
```

`--clean` 会删除项目的 `build/`、`python/build/`、`python/dist/` 和 egg-info，因此也会清除默认依赖缓存并重新下载。

### 构建环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `PYTHON_BIN` | `python3` | 创建构建和测试虚拟环境的解释器 |
| `DARTS_MESON_BUILD_DIR` | `build/meson` | Meson 构建目录 |
| `DARTS_DEPS_DIR` | `build/deps` | 本地依赖、sysroot 和下载缓存目录 |
| `DARTS_ONNXRUNTIME_DIR` | 自动下载目录 | ONNX Runtime 前缀，必须包含 `include/` 和 `lib/` |
| `DARTS_ONNXRUNTIME_VERSION` | `1.17.3` | 自动下载的 ONNX Runtime 版本 |
| `DARTS_ONNXRUNTIME_URL` | GitHub Release URL | 自定义 ONNX Runtime 下载地址或镜像 |
| `DARTS_BUILD_VENV` | `build/meson/.build-venv` | 构建虚拟环境目录 |
| `DARTS_TEST_VENV` | `build/test-venv` | wheel 验证虚拟环境目录 |
| `DARTS_DEBUG_MEMORY` | 未启用 | 设为 `1`、`true` 或 `yes` 时启用内存调试参数 |

示例：

```bash
PYTHON_BIN=/usr/bin/python3.12 \
DARTS_DEPS_DIR="$HOME/.cache/darts/deps" \
DARTS_ONNXRUNTIME_DIR=/opt/onnxruntime \
./scripts/build_all.sh --clean --test
```

如果网络需要代理，可沿用标准环境变量：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
./scripts/build_all.sh --clean --test
```

## 手动构建 wheel

如果依赖已经安装在系统 pkg-config 路径中，并且 ONNX Runtime 已准备好，可以直接使用 PEP 517 构建：

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip build

DARTS_ONNXRUNTIME_DIR=/opt/onnxruntime \
python -m build --wheel
```

`python -m build` 会创建隔离环境并安装 `setuptools`、`wheel`、Cython、Meson 和 Ninja。`setup.py` 随后调用 Meson、执行安装暂存、复制模型数据和运行库，并将扩展打入平台 wheel。

## 手动使用 Meson

仅编译原生库：

```bash
meson setup build/native . \
  -Dbuild-python=false \
  -Dbuild-tests=false \
  -Dlink-static=true \
  -Donnxruntime-dir=/opt/onnxruntime

meson compile -C build/native
```

同时编译 Python 扩展：

```bash
PYTHON_INCLUDE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"

meson setup build/native . \
  -Dbuild-python=true \
  -Dbuild-tests=false \
  -Dlink-static=true \
  -Donnxruntime-dir=/opt/onnxruntime \
  -Dpython-include-dir="$PYTHON_INCLUDE"

meson compile -C build/native
meson install -C build/native --destdir "$PWD/build/stage"
```

Meson 选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `build-python` | `false` | 构建 Cython 扩展 `cdarts` |
| `build-tests` | `false` | 构建 C++ 测试程序 |
| `link-static` | `true` | 优先静态链接支持该方式的依赖 |
| `debug-memory` | `false` | 启用 dmalloc 相关调试参数，需要额外安装 dmalloc |
| `onnxruntime-dir` | 空 | pkg-config 找不到 ONNX Runtime 时使用的前缀 |
| `python-include-dir` | 空 | 使用依赖 sysroot 时显式指定 Python 头文件目录 |

项目代码和生成的 protobuf 代码会静态合入 Python 扩展。发行版提供的 zlib、jsoncpp 和 protobuf 静态库通常没有启用 PIC，不能直接链接到 `.so`，因此 wheel 流程会动态链接并把对应运行库复制到 `darts/` 包目录。扩展的 RUNPATH 为 `$ORIGIN`，运行时会优先从自身目录加载这些库。

## 测试

推荐使用完整测试入口：

```bash
bash scripts/build_all.sh --test
```

该命令会启用 `build-tests=true`，依次运行：

- Meson 注册的 C++ 核心测试：AtomList、归一化、DAG 路径、词典编译/加载/匹配和配置分词。
- wheel 构建及隔离虚拟环境强制重装。
- Python `unittest` 黑盒测试：原子边界、归一化、词典标签及大小写、`faster`/`fast`、长文本、错误模式和多线程安全。

单独运行测试：

```bash
meson test -C build/meson --print-errorlogs
build/test-venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
```

已有构建虚拟环境依赖完整时，测试流程不会重复访问 Python 包索引，可离线重复执行。

## 安装 wheel

```bash
python3 -m venv .venv-runtime
source .venv-runtime/bin/activate
python -m pip install python/dist/darts-*.whl
```

验证安装：

```bash
python - <<'PY'
import darts
from darts import PyAtomList

atoms = PyAtomList("中文分词测试").tolist()
print("package:", darts.__file__)
print("atoms:", [(a.image, a.st, a.et, a.chtype) for a in atoms])
PY
```

wheel 与 Python ABI 绑定，例如 CPython 3.12 生成的 `cp312` wheel 不能安装到 CPython 3.11。需要为目标 Python 版本分别构建。

## Python 使用

### 原子切分和归一化

```python
from darts import PyAtomList, charDtype

alist = PyAtomList("Darts中文分词 123", skip_space=True, normal_before=True)
for atom in alist.tolist():
    print(atom.image, atom.st, atom.et, atom.chtype)

print(charDtype("中"))
```

`st` 和 `et` 表示归一化 Unicode 文本中的位置。`skip_space=True` 会过滤空白原子，`normal_before=True` 会在切分前执行归一化。

### 加载分词器

应用代码推荐使用高层 `Tokenizer`，接口兼容常见中文分词器调用方式：

```python
from darts import Tokenizer, cut, lcut, tokenize

tokenizer = Tokenizer("data/conf.json", "hybrid")
print(tokenizer.lcut("目标检测模型量化和中文分词测试"))
print(list(tokenizer.cut("目标检测模型量化")))
print(list(tokenizer.tokenize("中文ABC分词")))  # (词, 字符起点, 字符终点)
print(tokenizer(["第一句话", "第二句话"]))       # 批量调用

# 模块级函数延迟加载默认配置
print(lcut("中文分词"))
```

`cut_all=True` 或 `tokenize(..., mode="search")` 返回可能重叠的全部候选，不保证能拼成唯一切分；默认模式返回 Decider 选择的连续最佳路径。`HMM` 参数仅为调用兼容而保留，实际识别器由 `conf.json` 决定。需要 Atom、标签或候选图时再使用底层 `DSegment`：

```python
from darts import DSegment

segment = DSegment("data/conf.json", "fast")
atoms, words = segment.cut("目标检测模型量化和中文分词测试")

for word in words.tolist():
    print(word.image, word.atom_s, word.atom_e, sorted(word.labels))
```

模式为空字符串时使用配置中的 `default.mode`：

```python
segment = DSegment("data/conf.json", "")
```

返回的 `atom_s` 和 `atom_e` 是 AtomList 的半开区间。`labels` 来自词典标签、字符类型或序列识别器标签。

### 输出全部候选词

```python
atoms, candidates = segment.cut("中文分词", max_mode=True)
for word in candidates.tolist():
    print(word)
```

`max_mode=False` 使用 Decider 和最短路径输出单条最佳切分；`max_mode=True` 跳过路径决策，返回候选图中的全部词候选，主要用于调试和训练数据生成。

### 编码器

```python
from darts import AtomCodec, PyAtomList

codec = AtomCodec({"base.dir": "data/models/codex"}, "WordPice")
encoded = codec.encode(PyAtomList("Darts中文123"))
print(encoded)       # (token_code, atom_position)
print(codec.label_nums())
```

注意：注册类名在源码中是 `WordPice`，不是 `WordPiece`。

### 双数组模式匹配

```python
from darts import Dregex, PyAtomList, compileDregex

patterns = iter([
    ("北京大学", "ORG,SCHOOL"),
    ("自然语言处理", "NLP"),
])
compileDregex(patterns, "patterns.pb")

regex = Dregex("patterns.pb")
atoms = [a.image for a in PyAtomList("北京大学自然语言处理").tolist()]
regex.parse(atoms, lambda start, end, labels: print(start, end, labels))
```

### 句子级拼音标注

```python
from darts import PinyinAnnotator, sentence_pinyin

annotator = PinyinAnnotator("data/conf.json")
for token in annotator.annotate("重庆音乐ABC"):
    print(token.text, token.pinyin, token.start, token.end)

# 重庆 chóng qìng 0 2
# 音乐 yīn yuè 2 4
# ABC None 4 7
print(sentence_pinyin("银行行长前往上海"))
# yín háng háng zhǎng qián wǎng shàng hǎi
```

`PinyinAnnotator` 复用已加载的词典，适合连续处理句子。拼音模式优先使用词级短语读音消除多音字歧义，再回退到单字读音。非中文的 `pinyin` 默认是 `None`；可向 `annotate/readings` 传入 `non_cjk="<NO_PINYIN>"` 生成对齐占位符，或使用 `format(..., preserve_non_cjk=True)` 在格式化文本中保留原文。

### 中文地址与 POI 切分

```python
from darts import LocationSegmenter

location = LocationSegmenter("data/conf.json")
result = location.parse("浙江省杭州市西湖区文三路90号东部软件园2号楼")
for token in result.tokens:
    print(token.text, token.kind, token.start, token.end)
print(result.components)
```

`location` 模式使用独立的 `location.dict`、规则型 `AddressRecongnizer` 和概率型 `AddressDecider`。词典负责已知省、市、区县、乡镇街道及 POI；规则负责词典外道路/建筑后缀和数字门牌；量化器按行政层级与 POI 词性关系计算 `-log P(next_role | previous_role)`，地址 Bigram 只作为低权重词面证据。

重建数据和模型：

```bash
python scripts/data_pipeline.py download \
  --name china-administrative-divisions \
  --name china-administrative-divisions-license
python scripts/location_data.py
```

可在 `data/external/location/poi.txt` 中按每行一个 POI 名称追加业务或 OpenStreetMap 数据。全国 OSM PBF 较大且采用 ODbL 1.0，因此不在默认构建中自动下载；来源和许可要求见 `data/readme.md`。

## C API

公共 C API 位于 `src/main/darts.h`，主要生命周期如下：

```c
#include "darts.h"

init_darts_env();

segment sg = load_segment("data/conf.json", "fast", false);
atomlist atoms = asplit("中文分词", strlen("中文分词"), true, true);
wordlist words = token_str(sg, atoms, false);

/* 使用 walk_wlist 或 get_npos_word 读取结果 */

free_wordlist(words);
free_alist(atoms);
free_segment(sg);
destroy_darts_env();
```

由 API 创建的 `segment`、`atomlist`、`wordlist`、`dreg` 和编码器对象必须调用对应的 `free_*` 函数释放。Python 包装层会自动管理这些资源。

## 算法原理

### 1. 文本归一化与 AtomList

输入首先转换为 UTF-32 以稳定处理 Unicode 位置。归一化模块处理全角/半角和字符混淆等情况，原子切分模块再按字符类型生成 `Atom`：

- `image`：原子文本。
- `st`、`et`：在 Unicode 文本中的半开位置。
- `char_type`：中文、英文、数字、空白、标点等类型。
- `masked`：训练编码时使用的遮罩标记。

Atom 是候选词图的最小覆盖单位。

### 2. 候选词识别

`Segment::buildSegPath` 先为每个 Atom 添加一个单原子 Word，再顺序调用模式中配置的 Recognizer：

- `DictWordRecongnizer` 使用 Trie/双数组词典匹配连续 Atom，添加词典候选和标签。
- `OnnxRecongnizer` 枚举 2 到 `max.span` 个 Atom 的 span，使用 Transformer 独立预测每个 span 成词概率。候选可重叠，不使用 BIO、CRF、LSTM 或唯一切分约束；`threshold.N` 可按词长控制召回率与候选规模。
- `PinyinRecongnizer` 为候选添加拼音标签和编码；该插件设计为独占识别器。

候选可能重叠，例如“南京”“南京市”“市长”“长江”可以同时存在，后续决策器负责选择完整路径。

### 3. 候选图

`SegPath` 保存所有 Word 及其起止 Atom 位置。若前一候选的结束位置等于后一候选的开始位置，两者之间形成有向边；图中还包含起点和终点，因此每条完整路径都代表一种合法切分。

### 4. 边代价

Decider 分为嵌入和距离两个阶段：

- `embed()` 为候选词附加词典索引或神经网络向量。
- `ranging(pre, next)` 返回相邻候选的非负代价。

内置策略：

- `MinCoverDecider`：代价与词长成反比，偏向更长、数量更少的候选覆盖。
- `BigramDecider`：优先使用词文本查询 Bigram 索引，未登录词退化到词类型标签，再查询相邻词代价。
- `OnnxDecider`：使用 Transformer indicator 生成候选嵌入，再由归一化双塔 quantizer 批量计算相邻词关联概率的负对数。候选 embedding 共享连续缓冲区，避免逐词分配和复制。

### 5. 最优路径

候选按起点排序后构成 DAG，核心按拓扑顺序执行动态规划，在线性 `O(V + E)` 时间内计算从图起点到终点的最小总代价路径。最终按路径中的候选索引输出 `WordList`。

长文本会按标点和最大长度切成较短片段后分别解码，避免候选图无限增长。

## 架构模块

### 核心层 `src/core`

| 模块 | 职责 |
| --- | --- |
| `core.hpp` | `Atom`、`AtomList`、`Word`、`SegPath` 等核心数据结构 |
| `segment.hpp` | 插件接口、候选图构建、边生成、Dijkstra 和长文本切分 |

### 实现层 `src/impl`

| 模块 | 职责 |
| --- | --- |
| `confparser.hpp` | 读取 JSON、解析模式、递归加载插件依赖并缓存实例 |
| `recognizer.hpp` | 词典和拼音识别器 |
| `quantizer.hpp` | 最小覆盖与 Bigram 决策器 |
| `neuroplug.hpp` | ONNX Runtime 会话、序列识别器和神经图决策器 |
| `encoder.hpp` | WordPiece、标签和拼音编码器 |

### 工具层 `src/utils`

包含 UTF-8/UTF-32 转换、字符分类、归一化、拼音表、双数组 Trie、Bigram 存储、Protocol Buffers、文件资源定位和模式匹配等基础设施。

### API 层 `src/main` 与 `python/darts`

- `darts.h` 定义 C ABI。
- `darts.cxx` 将 C++ 对象包装为不透明句柄。
- `cdarts.pyx` 提供 Python 类和自动资源释放。
- `setup.py` 驱动 Meson 并组装二进制 wheel。

### 开发训练层 `python/darts/devel`

| 文件 | 职责 |
| --- | --- |
| `reader.py` | 文本、词候选和稀疏图训练数据构造 |
| `model.py` | 共享 Transformer 词表示架构、span 识别器和稀疏图量化器 |
| `utils.py` | `GraphLossSparse` 条件路径负对数似然 |
| `scripts/train_recognizer.py` | 独立训练和导出可重叠 span 识别器 |
| `scripts/train_quantizer.py` | 使用 `GraphLossSparse` 独立训练和导出图量化器 |
| `scripts/train_joint.py` | 推荐的共享 encoder 多任务训练与三个 ONNX 导出 |

开发模块依赖 PyTorch、NumPy 等训练库，这些大体积依赖不会随运行时 wheel 自动安装。

GPU 训练建议使用独立 Python 3.12 环境，并根据驱动和显卡选择 PyTorch 官方 CUDA wheel。当前 RTX A2000（`sm_86`）验证环境固定为 CUDA 12.8 构建：

```bash
python3.12 -m venv build/train-venv
build/train-venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1"
build/train-venv/bin/pip install numpy onnx onnxscript
build/train-venv/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())'
```

不要仅按本机安装的 CUDA Toolkit 版本选 wheel；以 NVIDIA 驱动兼容范围、PyTorch wheel 自带 CUDA runtime 和 GPU Compute Capability 为准。训练前必须确认 `torch.cuda.is_available()` 为真。

## 配置文件

默认配置位于 `data/conf.json`。配置由四部分组成：服务、识别器、决策器、运行模式。

```jsonc
{
  "dservices": {},
  "recognizers": {},
  "deciders": {},
  "modes": {},
  "default.mode": "faster"
}
```

配置解析器接受字符串参数。插件节点中的普通字符串字段会传给 `initalize()`；`deps` 中的值引用另一个插件实例，键则是当前插件期望的依赖参数名。

加载器会在三个插件区段中统一查找类型，通过注册工厂构造实例，并缓存共享依赖。空类型、错误依赖类型和循环依赖会在初始化阶段直接失败；依赖字段必须统一写作 `deps`。

### `dservices`

服务是可被 Recognizer 或 Decider 复用的编码组件。

| 注册类型 | 参数 | 说明 |
| --- | --- | --- |
| `LabelEncoder` | `hx.file` | 标签与权重映射文件，用于未登录词类型回退 |
| `WordPice` | `base.dir` | 包含 `engpiece.mbs` 和 `table.vocab` 的目录 |
| `PinyinEncoder` | 无 | 从内置拼音表构建编码空间 |

示例：

```jsonc
"dservices": {
  "wtype.encoder": {
    "type": "LabelEncoder",
    "hx.file": "data/codes/type.hx.txt"
  },
  "wordpiece.dict": {
    "type": "WordPice",
    "base.dir": "data/models/codex"
  }
}
```

### `recognizers`

| 注册类型 | 参数/依赖 | 说明 |
| --- | --- | --- |
| `DictWordRecongnizer` | `pbfile.path` | 编译后的词典文件 |
| `DictWordRecongnizer` | `atom.mode` | 字符串 `"true"` 时按每个标签生成独立候选 |
| `OnnxRecongnizer` | `model.path` | 输出各候选成词概率的 ONNX span 模型 |
| `OnnxRecongnizer` | `max.span` | 最大候选 Atom 长度，默认 `5` |
| `OnnxRecongnizer` | `threshold` / `threshold.N` | 全局及按长度覆盖的成词概率阈值 |
| `OnnxRecongnizer` | `deps.wordpiece.name` | 指向 `WordPice` 服务 |
| `OnnxSyntaxRecongnizer` | `model.path` / `label.path` | 输出 `NOT_WORD + POS` 多分类概率的 ONNX span 模型及标签表 |
| `OnnxSyntaxRecongnizer` | `max.span` / `threshold` | 最大候选长度及加入候选图的类别概率阈值 |
| `OnnxSyntaxRecongnizer` | `deps.wordpiece.name` | 指向 `WordPice` 服务 |
| `PinyinRecongnizer` | `deps.pyin.encoder` | 指向 `PinyinEncoder` 服务 |
| `PinyinRecongnizer` | `non-cjk.label` | 可选的非中文占位标签；为空时不添加拼音 |

示例：

```jsonc
"neural.span": {
  "type": "OnnxRecongnizer",
  "model.path": "data/models/neural/recognizer.onnx",
  "max.span": "5",
  "threshold.2": "0.45",
  "threshold.5": "0.65",
  "deps": {
    "wordpiece.name": "wordpiece.dict"
  }
}
```

### `deciders`

| 注册类型 | 参数/依赖 | 说明 |
| --- | --- | --- |
| `MinCoverDecider` | 无 | 使用词长启发式代价 |
| `BigramDecider` | `dat.path` | Bigram 二进制词典 |
| `BigramDecider` | `deps.type.enc` | 可选的 `LabelEncoder` 服务 |
| `HybridStatDecider` | `dat.path` | Bigram 二进制词典 |
| `HybridStatDecider` | `bigram.weight` | Bigram 负对数代价权重 |
| `HybridStatDecider` | `length.weight` / `length.power` | 词长先验的权重与幂次 |
| `HybridStatDecider` | `unknown.penalty` | 不在统计词表中的候选惩罚 |
| `HybridStatDecider` | `token.penalty` | 每个候选词的固定惩罚 |
| `OnnxDecider` | `pmodel.path` | indicator/embedding 模型路径 |
| `OnnxDecider` | `qmodel.path` | 边量化模型路径 |
| `OnnxDecider` | `deps.wordpiece.name` | WordPiece 服务依赖 |
| `OnnxDecider` | `deps.tencode.name` | 词类型编码服务依赖 |

`MinCoverDecider` 不需要统计模型，用于偏 precision 的 `faster` 模式；`BigramDecider` 用于兼容 `fast` 模式。默认 `hybrid` 使用 `HybridStatDecider`，将平滑 Bigram、词长先验和 OOV 惩罚组合为非负图边代价。词级概率采用插值 Kneser-Ney：根据出现一次/两次的 bigram 数量估计绝对折扣，未见搭配回退到 continuation probability，并融合较低权重的词尾/词首字符条件概率。中文 OOV 使用字符 bigram，英文使用字母字符模型与长度衰减，纯数字使用开放类长度模型。所有输出均为 `-log P`。参数由 `scripts/tune_decider.py` 在 dev 集搜索，禁止使用 test 集调参。

### Transformer 神经模型

识别器与量化器共享 `WordEncoder` 架构。推荐的联合训练让两个损失共同更新同一份 encoder 参数，识别概率头与关联 NLL 头保持独立；独立训练脚本仍用于消融。导出时共享参数分别固化到 `recognizer.onnx` 和 `indicator.onnx`，运行时不要求跨 ONNX 会话共享内存。编码器先生成带句内位置的上下文字表示，再用内容注意力和词内相对位置偏置对 span 内所有 WordPiece 加权组合，不采用简单首尾或平均池化。

识别器输出独立的 `word_probability`，允许同时召回交叠词。量化器输出 `association_nll = -log P(next | previous)`，使用 `GraphLossSparse` 在完整候选 DAG 上优化金标路径条件负对数似然。ONNX 使用 opset 18，并导出为 ONNX Runtime 1.17.3 可加载的 IR 9；项目不提供 CRF/BIO 兼容模型。

CUDA 联合训练默认关闭 PyTorch 的 flash/memory-efficient SDPA 反向内核并使用 math
SDPA。部分 Ampere 显卡在高度填充的变长批次上会由
`ScaledDotProductEfficientAttentionBackward` 产生非有限梯度；该设置只影响训练内核，
不会改变 Transformer 参数结构或 ONNX 推理模型。共享编码器同时关闭 PyTorch
Transformer 的原型 nested-tensor 快路径，继续使用标准稠密 padding mask。
量化器余弦归一化使用显式的有限范数下限，防止接近零的 key/query 投影在反向传播中
产生极大梯度。

### `modes`

模式把一个 Decider 和一组 Recognizer 组合为可加载流水线：

```jsonc
"modes": {
  "faster": {
    "decider": "mini.decider",
    "recognizers": ["inner.dict"],
    "temperature": 0.0,
    "random.seed": 42
  },
  "fast": {
    "decider": "ngram.decider",
    "recognizers": ["inner.dict"]
  }
},
"default.mode": "faster"
```

`DSegment(config, mode)` 中显式传入的 mode 优先；mode 为空时读取 `default.mode`。

`temperature` 控制完整分词路径的玻尔兹曼采样，默认 `0`，保持确定性最短路径。大于
`0` 时，路径概率与 `exp(-总 NLL / temperature)` 成正比；较小值偏向最优路径，较大值
会提高歧义切分的多样性。温度不强制限制在 `1` 以内，因为不同 Decider 的代价尺度不同；
可从 `0.5、1、2、5、10` 逐级观察候选多样性。`random.seed` 可选，用于复现实验。
Python 可按次覆盖配置：

```python
segment = DSegment("data/conf.json", "hybrid")
_atoms, words = segment.cut("南京市长江大桥", temperature=0.5)

tokenizer = Tokenizer(mode="hybrid")
alternative = tokenizer.sample("南京市长江大桥", temperature=0.5)
```

采样器在 DAG 上使用对数空间后向配分动态规划，时间复杂度为 `O(V + E)`，不会枚举
指数数量的完整路径。`temperature=0` 直接走原确定性路径算法，不引入随机数和配分计算。

默认配置提供 `hybrid`、`faster`、`fast`、`pinyin` 和 `neural`，默认选择 `hybrid`。`pinyin` 共享混合统计分词路径，再以词级短语拼音消除多音字歧义，未命中短语时回退到单字读音；非中文默认保持原分词标签且不添加拼音。`neural` 需要先生成 `data/models/neural/` 下的三个 ONNX 文件。

所有模式默认加载 `TemporalQuantityRecongnizer`。该规则识别器使用有界前视状态机，
不使用正则表达式，识别 `2026年7月16日`、`2026-07-16`、`July 16, 2026`、
`14:30:20`、`9点30分` 等中英文日期时间，以及 `12.5公斤`、`500ml`、
`2GB`、`三百二十个` 等数字量词和常见公制、时间、货币、存储单位。状态机直接在
AtomList 上运行，候选位置与训练、量化器和 `best_path` 使用同一 Atom 索引。

### LAC/POS 模式

`lac` 模式使用专用 `data/models/lac.pbs` 和 `atom.mode=true`。内部词性标签采用
`POS_*` 命名空间，主要接近 Universal POS，并保留 LAC/Jieba 常用的
`nr/ns/nt/nz/vn/q` 等细分类，避免与字符类型 `POS` 冲突。Python 可直接返回
LAC 风格短标签：

```python
from darts import Tokenizer

for token in Tokenizer(mode="lac").lac("中文分词在2026年发布"):
    print(token.text, token.pos, token.start, token.end)
```

训练语料支持兼容的 `词/POS_*` 空白分隔格式，例如：

```text
中文/POS_NOUN 分词/POS_NOUN 发布/POS_VERB
```

量化器训练图以 `(atom_start, atom_end, pos_type)` 标识节点，同一词面的多个候选
词性不会再按 span 合并。纯 CWS 空白分隔语料继续使用 unknown type，保持兼容。

### 语法标签识别器

`SyntaxSpanRecognizer` 对长度 `1..max.span` 的每个候选区间执行互斥多分类。类别
`0` 固定为 `NOT_WORD`，其余类别来自 `data/codes/pos.hx.txt`，例如
`POS_NOUN`、`POS_VERB`、`POS_PROPN`。模型一次推理同时判断候选是否成词以及成词
时的词法类型，不依赖 BIO 或 CRF，也不会强制候选区间互斥，仍可召回重叠歧义词。

```bash
PYTHON=/home/xuen/.venv/bin/python python3 scripts/devel.py syntax-train \
  --train data/generated/lac-train.txt \
  --dev data/generated/lac-dev.txt \
  --device cuda

PYTHON=/home/xuen/.venv/bin/python python3 scripts/devel.py syntax-export \
  model_bin/syntax/best.pt data/models/neural/syntax.onnx
```

导出同时生成 `syntax.labels.txt` 和 `syntax.json`。运行时配置示例：

```jsonc
"neural.syntax": {
  "type": "OnnxSyntaxRecongnizer",
  "model.path": "data/models/neural/syntax.onnx",
  "label.path": "data/models/neural/syntax.labels.txt",
  "max.span": "5",
  "threshold": "0.5",
  "deps": {
    "wordpiece.name": "wordpiece.dict"
  }
}
```

将 `neural.syntax` 加入具体 mode 的 `recognizers` 后才会实例化模型；默认配置仅声明
插件，不会在 ONNX 文件尚未导出时影响其他模式。

### 开发模式

C API 的 `load_segment(..., isdevel=true)` 或 Python 的 `DSegment(..., isdev=True)` 不加载 Decider。此时配合 `max_mode=True` 可以保留 Recognizer 产生的全部候选，主要用于训练图构建和调试。

## 运行时资源

资源查找按以下顺序进行：

1. 配置中给出的当前路径。
2. `cdarts` 动态库所在目录下的相对路径。
3. 动态库父目录下的相对路径。
4. `DARTS_CONF_PATH` 指向目录下的相对路径。

wheel 已将 `data/` 放入 `darts/` 包目录。默认的 `data/conf.json` 可由资源定位器从包目录解析。自定义部署可以这样组织：

```text
/srv/darts-runtime/
├── conf.json
├── models/
├── codes/
└── dictionaries/
```

```bash
export DARTS_CONF_PATH=/srv/darts-runtime
```

配置中的路径仍应相对于 `DARTS_CONF_PATH`，例如 `models/custom.onnx`。如果直接写绝对路径，则优先使用绝对路径。

## 数据文件

| 路径 | 用途 |
| --- | --- |
| `data/conf.json` | 默认插件与模式配置 |
| `data/kernel/chars.tmap` | 运行时字符类型映射，由人工规则与 FineFreq 高频字符生成 |
| `data/kernel/chars.manual.tmap` | 不应被频率数据覆盖的人工字符语义分类 |
| `data/kernel/confuse.json` | 归一化/易混字符映射 |
| `data/kernel/pinyin.txt` | 拼音数据 |
| `data/kernel/pinyin-phrases.txt` | 与核心词典对齐的词级拼音消歧数据 |
| `data/codes/type.hx.txt` | 词类型标签及权重 |
| `data/models/panda.pbs` | 内置词典 Trie |
| `data/models/ngram_dict.bdf` | Bigram 决策词典 |
| `data/models/neural/recognizer.onnx` | 可重叠 span 成词概率模型，由开发脚本生成 |
| `data/models/neural/indicator.onnx` | 量化器的独立词表示模型，由开发脚本生成 |
| `data/models/neural/quantizer.onnx` | 相邻词关联概率负对数模型，由开发脚本生成 |
| `data/models/codex/engpiece.mbs` | 英文子词词典 |
| `data/models/codex/table.vocab` | WordPiece 词表 |
| `data/demo/` | 神经训练、词典和模式编译示例数据 |

不要在不了解格式的情况下修改 `data/kernel/`。模型、词典、编码表和配置必须保持版本一致，否则可能出现模型输入维度、标签编号或 Trie 标签不匹配。

`chars.tmap` 可使用 FineFreq 的普通话与英文字符频率文件重建：

```bash
python scripts/devel.py tmap-build \
  --finefreq /path/to/FineFreq/csv/cmn_Hani.csv \
  --finefreq /path/to/FineFreq/csv/eng_Latn.csv
```

生成器默认仅选择总频次不低于 `100000` 的前 4096 个安全字符。人工映射优先，
但会把旧 `RUSH` 分类中的高频拉丁变音字纠正为 `ENG`，避免 `café` 一类单词在音标处
断开。FineFreq 只补齐拉丁扩展字母、十进制数字、标点和符号。CJK 字符不会写入映射，
仍由内置 Unicode CJK 区间逐字编码，避免高频汉字被合并成单个 Atom。其他文字系统
保持 `WUNK`，不会因为英文网页中的噪声字符改变默认语言边界。

## 开发工具

统一开发入口为 `scripts/devel.py`：

```bash
# 将 LABELS:WORD 文本编译为紧凑 v2 词典
python scripts/devel.py dict-compile data/demo/dregex_pattern_file.txt /tmp/demo.pbs

# 无需原始文本，将旧词典转换为 v2；先执行一次 Meson 构建
python scripts/devel.py dict-repack data/models/panda.pbs /tmp/panda-v2.pbs

# 编译 Kneser-Ney + 字符/OOV 回退统计模型
python scripts/devel.py bigram-compile \
  data/generated/bigram-single.txt \
  data/generated/bigram-union.txt \
  data/models/ngram_dict.bdf

# 测量词典加载和匹配吞吐
python scripts/devel.py dict-benchmark /tmp/panda-v2.pbs --repeat 10000

# 独立训练 span 识别器；默认枚举 2~5 Atom，CUDA 可用时自动使用 GPU
python scripts/devel.py model-train data/generated/cws-train.txt --epochs 20 \
  --max-span 5 --output-dir model_bin/recognizer

# 导出成词概率 ONNX 及包含各长度阈值的 JSON 元数据
python scripts/devel.py model-export model_bin/recognizer/best.pt \
  data/models/neural/recognizer.onnx

# 独立训练 GraphLossSparse 量化器并导出词表示网络和关联 NLL 网络
python scripts/devel.py quantizer-train --epochs 20 --output-dir model_bin/quantizer
python scripts/devel.py quantizer-export model_bin/quantizer/best.pt data/models/neural

# 推荐：一次多任务训练共享 encoder，仍分别导出三个运行时 ONNX
python scripts/devel.py joint-train --epochs 20 --output-dir model_bin/joint
python scripts/devel.py joint-evaluate model_bin/joint/best.pt \
  --data data/generated/cws-test.txt
python scripts/devel.py joint-export model_bin/joint/best.pt data/models/neural

# RTX A2000 等旧架构在新 CUDA wheel 上若出现 AMP 内核错误，可关闭 AMP 并续用已有权重
python scripts/devel.py joint-train --no-amp --resume model_bin/joint/best.pt \
  --epochs 20 --output-dir model_bin/joint

# 先用小型格式样本做一轮 smoke test（不用于准确率报告）
python scripts/train_joint.py train --train data/demo/cws-train.txt \
  --dev data/demo/cws-dev.txt --epochs 1 --hidden-size 32 \
  --output-dir /tmp/darts-joint-demo

# 下载开放数据、生成训练切分并重建词典/Bigram 模型
python scripts/devel.py data download
python scripts/devel.py data prepare
python scripts/devel.py data build-models

# 执行纯 Meson 构建和 wheel 回归
python scripts/devel.py build --test
```

### 开放数据流水线

`data/sources.json` 固定 UD Chinese GSD 2.18 与 jieba 0.42.1 的下载地址和许可；`data/sources.lock.json` 记录实际文件大小和 SHA-256。原始文件下载到被 Git 忽略的 `data/external/`，训练切分和中间文本生成到 `data/generated/`。下载器继承 `HTTP_PROXY`/`HTTPS_PROXY`，支持 `.part` 断点续传和已有文件校验。

```bash
python scripts/data_pipeline.py download
python scripts/data_pipeline.py prepare
python scripts/data_pipeline.py build-models
python scripts/data_pipeline.py evaluate data/generated/cws-dev.txt --mode faster
python scripts/data_pipeline.py evaluate data/generated/cws-test.txt --mode faster

# 仅在开发集扫描混合统计量化器参数
python scripts/tune_decider.py
python scripts/tune_decider.py --fine
```

词典保留全部 UD train 词汇，并合并 jieba 中频率至少为 10 的词条；默认过滤超过 8 个 Unicode 字符的异常长词。Bigram 只从 UD train 统计，dev 用于模式选择，test 仅用于最终评测。UD Chinese GSD 使用 CC BY-SA 4.0，jieba 词典使用 MIT；再分发时必须遵守各自署名与共享条款，详见 `data/readme.md` 和下载后的许可证文件。

在 UD Chinese GSD dev 的 500 句上，调优后的 `faster` 模式边界 precision 为 0.8856、recall 为 0.9434、F1 为 0.9136；历史内置模型同模式 F1 为 0.7993。该结果是单一语料域基线，不代表新闻、医疗或社交文本的泛化质量。

混合统计量化器在 dev 上取得 precision 0.8830、recall 0.9568、F1 0.9185；未参与调参的 test 上取得 precision 0.8634、recall 0.9436、F1 0.9017、整句准确率 0.112。相同 test 上 `faster` 的 F1 为 0.8971、整句准确率为 0.094。若业务更重视 precision，可显式选择 `faster`。

词典 v2 使用差分 ZigZag 状态数组和最高级别 Deflate 压缩，同时保留旧 `.pbs` 的读取兼容。最高压缩会增加词典生成时间，但不影响运行时识别速度。`darts-dict-repack` 要求输入和输出路径不同，以免损坏原文件。

`DictWordRecongnizer` 对连续 Atom 使用专用 Aho-Corasick 扫描，不再构造位置环形缓冲或经过虚函数回调；ASCII 大写仅在确实需要时执行大小写折叠。通用 C/Python `Dregex.parse()` 接口继续保留原迭代器语义。

## 自定义模式示例

只使用自定义词典和最小覆盖策略：

```jsonc
{
  "dservices": {},
  "recognizers": {
    "custom.dict": {
      "type": "DictWordRecongnizer",
      "pbfile.path": "models/custom.pbs",
      "atom.mode": "false"
    }
  },
  "deciders": {
    "min.cover": {
      "type": "MinCoverDecider"
    }
  },
  "modes": {
    "custom": {
      "decider": "min.cover",
      "recognizers": ["custom.dict"]
    }
  },
  "default.mode": "custom"
}
```

启动前确保 `models/custom.pbs` 能通过资源查找规则定位。

## 常见问题

### Meson 找不到 Cython

使用一键脚本，或在当前环境安装构建工具：

```bash
python -m pip install "Cython>=3.0" "meson>=1.3" "ninja>=1.11"
```

### 找不到 ONNX Runtime

```bash
export DARTS_ONNXRUNTIME_DIR=/opt/onnxruntime
test -f "$DARTS_ONNXRUNTIME_DIR/include/onnxruntime_cxx_api.h"
ls "$DARTS_ONNXRUNTIME_DIR/lib"/libonnxruntime.so*
```

### `Python.h: No such file or directory`

安装目标 Python 版本的开发包，并显式传入 include 目录：

```bash
sudo apt install python3-dev
python -c 'import sysconfig; print(sysconfig.get_paths()["include"])'
```

### wheel 安装后缺少动态库

检查扩展：

```bash
python - <<'PY'
import pathlib, darts
print(pathlib.Path(darts.__file__).parent)
PY

ldd /path/to/site-packages/darts/cdarts*.so
```

正常情况下 `libonnxruntime`、`libprotobuf`、`libjsoncpp` 和 `libz` 应解析到同一个 `darts/` 包目录。

### 配置或模型找不到

优先使用绝对配置路径，或设置资源根目录：

```bash
export DARTS_CONF_PATH=/srv/darts-runtime
```

### 模式加载失败

检查以下内容：

- `modes.<name>.decider` 是否存在于 `deciders`。
- `modes.<name>.recognizers` 中每项是否存在于 `recognizers`。
- `deps` 的值是否指向已定义插件。
- `type` 是否与源码注册名称完全一致，包括现有拼写。
- 模型和词典路径是否能由资源定位器找到。

### 性能验证

完成构建并安装 wheel 后，可运行可重复基准：

```bash
python scripts/benchmark_runtime.py --iterations 1000
```

脚本分别测量原子化及 `faster`、`fast` 两种分词模式，输出总耗时、每次调用耗时和吞吐量。候选词在识别完成后按起点建立索引；分词图使用连续邻接表和 DAG 动态规划；支持批量接口的 ONNX 决策器会一次计算全部候选边，旧版一维 ONNX 模型则自动回退到逐边推理。

## 构建验证状态

当前自动化流程已验证以下链路：

- Linux x86_64。
- GCC 15 / C++17。
- Meson 1.11。
- Cython 3.2。
- CPython 3.14。
- ONNX Runtime 1.17.3。
- 从清理、依赖下载、原生编译到 wheel 生成。
- 在全新虚拟环境安装 wheel。
- `import darts`、`PyAtomList` 和 AtomList 基础调用。
- `faster`、`fast` 模式的完整 `DSegment` 推理。
- wheel 内动态库通过 `$ORIGIN` 加载。

其他 Python 版本原则上受 `python_requires >= 3.8` 支持，但应在目标版本上单独构建并执行 `./scripts/build_all.sh --test`。

## License

Apache License 2.0，详见 `LICENSE`。

## Contact

nanhangxuen@163.com
