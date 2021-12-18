# darts
a chinese fast smart tokenizer 

## Introduction

`C++` headers(hpp) with Python style. 

## Feature

+ linking is a annoying thing, so I write these source code in headers file(`*.hpp`), you can use them only with `#include "xx.hpp"`, without linking *.a or *.so .

**`no linking , no hurts`** 

## Example

See Details in `test/demo.cpp`

## Cases

1. [CppJieba]

## Reference

1. 提供基本的字符串归一化功能
2. 提供字符串按照字符类型的基本切分功能
3. 提供中文转拼音功能
4. 提供拼音自动切分
5. 提供字符串职能切分

6.  `md5.hpp` is copied from network, you can find original author in source code(in comments).
7.  `MutexLock.hpp`, `BlockingQueue.hpp`, `Condition.hpp` reference from [muduo].

## Contact

+ i@yanyiwu.com

[CppJieba]:https://github.com/yanyiwu/cppjieba.git
[muduo]:https://github.com/chenshuo/muduo.git