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
#include <stdbool.h>
#include <stdlib.h>

typedef void* ext_data;
typedef struct _dregex* dreg;

typedef struct _atom* atom;
typedef struct _decider* decider;
typedef struct _atomlist* atomlist;
typedef struct _wordlist* wordlist;
typedef struct _segment* segment;
typedef struct _encoder* encoder;

/**
 * @brief init so env
 *
 */
void init_darts_env();

/**
 * @brief  free darts env
 *
 */
void destroy_darts_env();

/**
 * @brief normalize_str
 *
 * @param str
 * @return char*
 */
char* normalize_str(const char* str, size_t len, size_t* ret);
/**
 * @brief 某个词的词类型
 *
 * @param word
 * @return char* 某个字符的字符类型
 */
const char* word_type(const char* word, size_t len);

/**
 * @brief load the drgex from file
 *
 * @param path
 * @param regex
 * @return int
 */
dreg load_dreg(const char* path);
void free_dregex(dreg regex);

/**
 * @brief  正则匹配
 *
 * @param regex
 * @param atomlist
 * @param len
 * @param ret
 * @return int,匹配的结果数，如果-1失败
 */

typedef bool (*atom_iter)(const char** word, size_t* postion, ext_data user_data);
typedef bool (*kv_iter)(const char** key, size_t* keylen, const char** labels, size_t* label_nums, size_t max_key_len,
                        size_t max_lables_len, ext_data user_data);
typedef bool (*dregex_hit)(size_t s, size_t e, const char** labels, size_t labels_num, ext_data user_data);

/**
 * @brief  parse a atom list
 *
 * @param regex
 * @param atomlist
 * @param user_data
 * @return int
 */
void parse(dreg regex, atom_iter atomlist, dregex_hit hit, ext_data user_data);
/**
 * @brief compile regex
 *
 * @param outpath
 * @param kvs
 * @param user_data
 * @return int
 */
int compile_regex(const char* outpath, kv_iter kvs, ext_data user_data);

atomlist asplit(const char* txt, size_t textlen);
void free_alist(atomlist alist);
segment load_segment(const char* conf_file, const char* mode);
void free_segment(segment sg);
wordlist token_str(segment sg, atomlist alist, bool max_mode, bool normal_before);
void free_wordlist(wordlist wlist);

#ifdef __cplusplus
}
#endif  /* __cplusplus */
#endif  // SRC_MAIN_DARTS_H_
