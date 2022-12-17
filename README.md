# darts


中文分词框架(a chinese fast smart tokenizer)本分词程序采用自定义神经网络结构结合常规规则等算法，折中考虑了效果与性能的分词程序☕️ 用户可以自定义词典、模型、规则来决定分词器的基本行为，并且可以修改json配置文件
来决定具体的行为


## Introduction

`C++` headers(hpp) with Python style. 

## Install

```bash
1. pip install darts.whl
2. if you need use devel package you must install depence package yourself
```

## Compile

```bash
1. 编译安装需要安装依赖，具体依赖参见cmakelist文件或者meson文件
2. 建议全部依赖都必须有静态依赖包同时开启了-fPIC 或则set(CMAKE_POSITION_INDEPENDENT_CODE ON) ,这样生成的so文件可以更加方便的运行
3. 使用cmake或者meson进行安装
4. 进入python目录编译python的安装包文件

```

## Feature

+ linking is a annoying thing, so I write these source code in headers file(`*.hpp`), you can use them only with `#include "xx.hpp"`, without linking *.a or *.so .

**`no linking , no hurts`** 

## Example

See Details in `test/demo.cpp`

## Cases

1. [Darts]

## Reference

1. 提供基本的字符串归一化功能
2. 提供字符串按照字符类型的基本切分功能
3. 提供中文转拼音功能
4. 提供拼音自动切分
5. 提供字符串职能切分
6. 提供中文地址规范化工具
7. 提供面向c++/python等语言的基础API
8. 提供修改python的模型开发工具



## Contact

nanhangxuen@163.com
