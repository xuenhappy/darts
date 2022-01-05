cdef extern from 'darts.h':    
    void init_darts()
    void destroy_darts()
    ctypedef void* darts_ext
    ctypedef struct _dregex* dregex
    ctypedef struct _segment* segment
    int normalize_str(const char* str, char** ret)
    int load_drgex(const char* path, dregex* regex)
    void free_dregex(dregex regex)
    ctypedef bool (*atom_iter)(const char** word, size_t* postion, darts_ext user_data)
    ctypedef bool (*dregex_hit)(size_t s, size_t e, const char** labels, size_t labels_num, darts_ext user_data)
    void parse(dregex regex, atom_iter atomlist, dregex_hit hit, darts_ext user_data)
    int load_segment(const char* json_conf_file, segment* sg)
    ctypedef void (*word_hit)(const char* strs, const char* label, size_t ast, size_t aet, size_t wst, size_t wet,darts_ext user_data)
    void token_str(segment sg, const char* txt, word_hit hit, bool max_mode, darts_ext user_data)
    void free_segment(segment sg)
    int word_type(const char* word, char** ret)
    ctypedef bool (*token_hit)(const char* strs, const char* label, size_t s, size_t e, darts_ext user_data)
    void word_split(const char* strs, token_hit hit, darts_ext user_data)
    void word_bpe(const char* strs, token_hit hit, darts_ext user_data)
