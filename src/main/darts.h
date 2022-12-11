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
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <stdbool.h>
#include <stdlib.h>

typedef struct _dregex* dreg;
typedef struct _decider* decider;
typedef struct _atomlist* atomlist;
typedef struct _wordlist* wordlist;
typedef struct _segment* segment;
typedef struct _encoder* encoder;

// first do
void init_darts_env();
// last do
void destroy_darts_env();

// normalize a str
char* normalize_str(const char* str, size_t len, size_t* ret);
// a unicode char type
const char* chtype(const char* word);

typedef struct {
    const char* image;  // image of atom
    size_t st;          // start of this atom in str
    size_t et;          // end of this atom in str
    const char* char_type;
    bool masked;  // this is f
} atom_;
typedef bool (*walk_alist_hit)(void* user_data, atom_* ret);
// convert a text to a alist
atomlist asplit(const char* txt, size_t textlen, bool skip_space, bool normal_before);
// free the atomlist
void free_alist(atomlist alist);
// walk alist
void walk_alist(atomlist alist, walk_alist_hit hit, void* user_data);

// give the alist len
size_t alist_len(atomlist alist);

// load the dregex from file
dreg load_dregex(const char* path);
// free the dregex
void free_dregex(dreg regex);

typedef struct {
    const char* word;
    size_t len;
    size_t postion;
} atomiter_ret;

typedef bool (*atomiter)(void* user_data, atomiter_ret* ret);

typedef struct {
    size_t s;
    size_t e;
    const char** labels;
    size_t labels_size;
} dhit_ret;

typedef bool (*dhit)(void* user_data, dhit_ret* ret);
typedef struct {
    const char** key;
    size_t keylen;
    const char** labels;
    size_t label_nums;
} kviter_ret;
typedef bool (*kviter)(void* user_data, kviter_ret* ret);

// parse a atom list
void parse(dreg regex, atomiter atomlist, dhit hit, void* user_data);
// compile regex
int compile_regex(const char* outpath, kviter kvs, void* user_data);
// load segment
segment load_segment(const char* conffile, const char* mode, bool isdevel);
// free segment
void free_segment(segment sg);
// token str
wordlist token_str(segment sg, atomlist alist, bool max_mode);
// free list
void free_wordlist(wordlist wlist);

#ifdef __cplusplus
}
#endif  /* __cplusplus */
#endif  // SRC_MAIN_DARTS_H_
