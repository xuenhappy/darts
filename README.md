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

> 当前源码的 `src/main/darts.cxx::load_segment()` 在成功分支错误地返回了
> `EXIT_SUCCESS`，而不是新建的 `segment` 句柄。因此本节展示的是该返回值缺陷修复后的
> API 用法；当前版本直接构造 `DSegment` 会抛出 `OSError`。AtomList、字符类型和编码等
> 不经过该入口的 API 不受影响。

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
- `OnnxRecongnizer` 使用 WordPiece 编码输入 ONNX 序列标注模型，根据 `B/I/O` 标签合并新词候选。
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
- `OnnxDecider`：使用 ONNX indicator 生成候选嵌入，再由 quantizer 模型计算边代价。

### 5. 最优路径

核心使用 Dijkstra 算法计算从图起点到终点的最小总代价路径。由于 `ranging()` 要求返回非负值，Dijkstra 的假设成立。最终按路径中的候选索引输出 `WordList`。

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
| `model.py` | 编码器、量化器和 GraphTrainer |
| `utils.py` | 图损失、稀疏图损失和训练工具 |
| `trainer.py` | 训练流程 |
| `export.py` | 模型导出 |
| `sover.py` | 辅助求解逻辑 |

开发模块依赖 PyTorch、NumPy 等训练库，这些大体积依赖不会随运行时 wheel 自动安装。

## 配置文件

默认配置位于 `data/conf.json`。配置由四部分组成：服务、识别器、决策器、运行模式。

```jsonc
{
  "dservices": {},
  "recognizers": {},
  "deciders": {},
  "modes": {},
  "default.mode": "fast"
}
```

配置解析器接受字符串参数。插件节点中的普通字符串字段会传给 `initalize()`；`deps` 中的值引用另一个插件实例，键则是当前插件期望的依赖参数名。

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
| `OnnxRecongnizer` | `model.path` | ONNX 序列标注模型 |
| `OnnxRecongnizer` | `label.list` | 逗号分隔的输出标签列表 |
| `OnnxRecongnizer` | `deps.wordpice.name` | 指向 `WordPice` 服务，键名沿用源码拼写 |
| `PinyinRecongnizer` | `deps.pyin.encoder` | 指向 `PinyinEncoder` 服务 |

示例：

```jsonc
"hmm.new.finder": {
  "type": "OnnxRecongnizer",
  "label.list": "O,B-_HWORD,I-_HWORD",
  "model.path": "data/models/crf.ner.onnx",
  "deps": {
    "wordpice.name": "wordpiece.dict"
  }
}
```

### `deciders`

| 注册类型 | 参数/依赖 | 说明 |
| --- | --- | --- |
| `MinCoverDecider` | 无 | 使用词长启发式代价 |
| `BigramDecider` | `dat.path` | Bigram 二进制词典 |
| `BigramDecider` | `deps.type.enc` | 可选的 `LabelEncoder` 服务 |
| `OnnxDecider` | `pmodel.path` | indicator/embedding 模型路径 |
| `OnnxDecider` | `qmodel.path` | 边量化模型路径 |
| `OnnxDecider` | `deps.wordpiece.name` | WordPiece 服务依赖 |
| `OnnxDecider` | `deps.tencode.name` | 词类型编码服务依赖 |

`MinCoverDecider` 不需要模型，适合验证词典候选；`BigramDecider` 是默认 `fast` 模式使用的稳定决策器。

### `modes`

模式把一个 Decider 和一组 Recognizer 组合为可加载流水线：

```jsonc
"modes": {
  "faster": {
    "decider": "mini.decider",
    "recognizers": ["inner.dict"]
  },
  "fast": {
    "decider": "ngram.decider",
    "recognizers": ["inner.dict", "hmm.new.finder"]
  }
},
"default.mode": "fast"
```

`DSegment(config, mode)` 中显式传入的 mode 优先；mode 为空时读取 `default.mode`。

当前仓库的 `faster` 和 `fast` 模式有完整数据文件。`smart` 配置中的 `data/model/onnx/` 模型目录未包含在当前仓库中；`pinyin` 模式引用的决策器名称也没有对应完整注册配置，因此这两个模式需要补充模型或修正插件配置后才能用于生产。

### 开发模式

C API 的 `load_segment(..., isdevel=true)` 或 Python 的 `DSegment(..., isdev=True)` 不加载 Decider。此时配合 `max_mode=True` 可以保留 Recognizer 产生的全部候选，主要用于训练图构建和调试。

## 运行时资源

资源查找按以下顺序进行：

1. 配置中给出的当前路径。
2. `cdarts` 动态库所在目录下的相对路径。
3. 动态库父目录下的相对路径。
4. `DARTS_CONF_PATH` 指向目录下的相对路径。

wheel 已将 `data/` 放入 `darts/` 包目录。修复上述 `load_segment()` 返回值缺陷后，默认的 `data/conf.json` 可由资源定位器从包目录解析。自定义部署可以这样组织：

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
| `data/kernel/chars.tmap` | 字符类型映射 |
| `data/kernel/confuse.json` | 归一化/易混字符映射 |
| `data/kernel/pinyin.txt` | 拼音数据 |
| `data/codes/type.hx.txt` | 词类型标签及权重 |
| `data/models/panda.pbs` | 内置词典 Trie |
| `data/models/ngram_dict.bdf` | Bigram 决策词典 |
| `data/models/crf.ner.onnx` | 新词序列识别模型 |
| `data/models/codex/engpiece.mbs` | 英文子词词典 |
| `data/models/codex/table.vocab` | WordPiece 词表 |
| `data/demo/` | 词典和模式编译示例数据 |

不要在不了解格式的情况下修改 `data/kernel/`。模型、词典、编码表和配置必须保持版本一致，否则可能出现模型输入维度、标签编号或 Trie 标签不匹配。

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
- 当前版本是否已经修复 `src/main/darts.cxx` 中 `load_segment()` 成功分支返回 `EXIT_SUCCESS` 的问题。

### `DSegment` 总是抛出 `OSError`

当前版本存在一个已确认的 C API 返回值缺陷：`load_segment()` 创建句柄后返回了空值。修复时应将成功分支末尾的返回值改为新建的 `sg` 句柄，并重新构建 wheel。该问题不代表 JSON 或模型一定加载失败。

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
- wheel 内动态库通过 `$ORIGIN` 加载。

完整 `DSegment` 推理尚未通过验证，原因是上文记录的 `load_segment()` 返回值缺陷。

其他 Python 版本原则上受 `python_requires >= 3.8` 支持，但应在目标版本上单独构建并执行 `./scripts/build_all.sh --test`。

## License

Apache License 2.0，详见 `LICENSE`。

## Contact

nanhangxuen@163.com
