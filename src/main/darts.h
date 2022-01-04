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
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <stdlib.h>


typedef void* darts_ext;
typedef struct _dregex* dregex;
typedef struct _segment* segment;


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

typedef bool (*atom_iter)(char* word, size_t postion, darts_ext user_data);
typedef bool (*dregex_hit)(size_t s, size_t e, int64_t* labels, size_t labels_num, darts_ext user_data);

/**
 * @brief  parse a atom list
 *
 * @param regex
 * @param atomlist
 * @param user_data
 * @return int
 */
int parse(dregex regex, atom_iter atomlist, dregex_hit hit, darts_ext user_data);


/**
 * @brief load the dregex from json conf file
 *
 * @param json_conf_file
 * @param sg
 * @return int
 */
int load_segment(const char* json_conf_file, segment* sg);

/**
 * @brief token匹配函数
 *
 */
typedef bool (*token_hit)(const char* str, const char* label, size_t s, size_t e, darts_ext user_data);
/**
 * @brief 分词
 *
 * @param sg
 * @param txt
 * @param len
 * @param ret
 * @return int 分词的结果数，如果-1失败
 */
void tokenize(segment sg, const char* txt, size_t len, token_hit hit, bool max_mode, darts_ext user_data);

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
 * @return char* 某个字符的字符类型
 */
char* word_type(const char* word);


/**
 * @brief char list
 *
 * @param str
 * @param len
 * @param hit
 * @param user_data
 */
void word_split(const char* str, size_t len, token_hit hit, darts_ext user_data);

#ifdef __cplusplus
}
#endif  /* __cplusplus */
#endif  // SRC_MAIN_DARTS_H_
