/*
 * File: darts.h
 * Project: main
 * File Created: Tuesday, 4th January 2022 5:39:25 pm
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Tuesday, 4th January 2022 5:39:28 pm
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */

#ifndef SRC_MAIN_DARTS_H_
#define SRC_MAIN_DARTS_H_
#include <stdlib.h>
extern "C" {
typedef struct _dregex* dregex;

typedef struct _segment* segment;

typedef struct _atom {
    char* word;
    size_t postion;
} * atom;


typedef struct _dergex_group {
    size_t s, e;
    int64_t* labels;
    size_t labels_num;
} * dergex_group;

typedef struct _token {
    char* word;
    size_t s, e;
    char* type;
} * token;

/**
 * @brief normalize ori string
 *
 * @param str
 * @param len
 * @return char*
 */
char* normalize_str(const char* str, size_t len);
/**
 * @brief load the drgex from file
 *
 * @param path
 * @param regex
 * @return int
 */
int load_drgex(const char* path, dregex* regex);

/**
 * @brief free memory
 *
 * @param regex
 */
void free_dregex(dregex regex);


/**
 * @brief  正则匹配
 *
 * @param regex
 * @param atomlist
 * @param len
 * @param ret
 * @return int,匹配的结果数，如果-1失败
 */
int parse(dregex regex, const atom* atomlist, size_t len, dergex_group* ret);

/**
 * @brief
 *
 * @param group
 */
void free_dregex_ret(dergex_group* group, size_t len);

/**
 * @brief
 *
 * @param atomlist
 * @param size
 */
void free_atomlist(atom* atomlist, size_t size);

/**
 * @brief load the dregex from json conf file
 *
 * @param json_conf_file
 * @param sg
 * @return int
 */
int load_segment(const char* json_conf_file, segment* sg);

/**
 * @brief 分词
 *
 * @param sg
 * @param txt
 * @param len
 * @param ret
 * @return int 分词的结果数，如果-1失败
 */
int tokenize(segment sg, const char* txt, size_t len, token* ret, bool max_mode);

/**
 * @brief free segment
 *
 * @param sg
 */
void free_segment(segment sg);


/**
 * @brief 某个词的词类型
 *
 * @param word
 * @return char*
 */
char* word_type(const char* word);

/**
 * @brief 对原始的
 *
 * @param str
 * @param len
 * @param ret
 * @return int 返回单词的结果数
 */
int word_split(const char* str, size_t len, atom* ret);
}
#endif  // SRC_MAIN_DARTS_H_
