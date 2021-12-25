# darts


中文分词框架(a chinese fast smart tokenizer)
本分词程序采用自定义神经网络结构结合常规规则等算法，折中考虑了效果与性能的分词程序

## Introduction

`C++` headers(hpp) with Python style. 

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
6. 提供面向c++/python等语言的基础API
7. 提供修改python的模型开发工具



## Contact

nanhangxuen@163.com
