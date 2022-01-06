from libc.stdlib cimport malloc, free
from libcpp cimport bool


cdef extern from 'darts.h': 
    ctypedef void* dregex
    ctypedef void* segment
    ctypedef void* darts_ext
    ctypedef bool (*atom_iter)(const char** word, size_t* postion, darts_ext user_data)
    ctypedef bool (*dregex_hit)(size_t s, size_t e, const char** labels, size_t labels_num, darts_ext user_data)
    ctypedef void (*word_hit)(const char* str, const char* label, size_t as, size_t ae, size_t ws, size_t we,darts_ext user_data)
    ctypedef bool (*token_hit)(const char* str, const char* label, size_t s, size_t e, darts_ext user_data)

    void init_darts()
    void destroy_darts()
    int normalize_str(const char* str, char** ret)
    int load_drgex(const char* path, dregex* regex)
    void free_dregex(dregex regex)
    void parse(dregex regex, atom_iter atomlist, dregex_hit hit, darts_ext user_data)
    int load_segment(const char* json_conf_file, segment* sg)
    void token_str(segment sg, const char* txt, word_hit hit, bool max_mode, darts_ext user_data)
    void free_segment(segment sg)
    int word_type(const char* word, char** ret)
    void word_split(const char* strs, token_hit hit, darts_ext user_data)
    void word_bpe(const char* strs, token_hit hit, darts_ext user_data)


#init some thing
init_darts()

def normalize(content):
    if content is None:
        return None
    content = content.encode('utf-8','replace')
    cdef char* point=NULL
    cdef size=normalize_str(content,&point)
    try:
        pyStr=point[:size].decode('utf-8','replace')
    finally:
        free(point)


#define some callback function
cdef bool atom_iter_func(const char** word, size_t* postion, darts_ext user_data):
    atom_info=next((<object>user_data)[0])
    if(atom_info is None):
        return False
    atom, idx=atom_info
    word[0]=<char*>atom
    postion[0]=<size_t>idx
    return True

cdef bool dregex_hit_callback (size_t s, size_t e, const char** labels, size_t labels_num, darts_ext user_data):
    py_hit=(<object>user_data)[1]
    labels_py=[]
    if(labels and labels_num>0):
        labels_py=[labels[i].decode("utf-8","ignore") for i in range(labels_num) if labels[i]]
    py_hit(s,e,labels_py)
    return False


cdef class Dregex:
    cdef dregex data

    def __cinit__(self, path):
        assert path is not None,"path must be give"
        self.data=NULL
        load_drgex(path.encode("utf-8",'ignore'), &self.data)
        if not self.data:
            raise IOError("load %s regex file failed!"%path)

    def parse(self, atom_iter,hit):
        user_data =(atom_iter,hit)
        parse(self.data,atom_iter_func,dregex_hit_callback, <darts_ext>user_data)
        

    def __dealloc__(self):
        free_dregex(self.data)
        self.data=NULL
        

class Token:
    def __init__(self):
        self.word=None
        self.start_of_string=0
        self.end_of_string=0
        self.start_of_atoms=0
        self.end_of_atoms=0
        self.tags=None

cdef void word_hit_callback(const char* strs, const char* label, size_t ast, size_t aet, size_t wst, size_t wet,darts_ext user_data):
    tokens=<object>user_data
    t=Token()
    if strs:
        t.word=strs.decode("utf-8",'ignore')
    if label:
        t.tags=label.decode("utf-8",'ignore')
    t.start_of_string=ast
    t.end_of_string=aet
    t.start_of_atoms=wst
    t.end_of_atoms=wet
    tokens.append(t)


cdef class Segment:
    cdef segment sg

    def __cinit__(self, path=None):
        if path is not None:
            path="data/darts.conf.json"
        self.sg=NULL
        load_segment(path.encode("utf-8",'ignore'), &self.sg)
        if not self.sg:
            raise IOError("load %s segment conf json file failed!"%path)

    def tokenize(self,strs,max_mode=False):
        if not strs:
            return []
        tokens=[]
        token_str(self.sg, strs.encode('utf-8','replace'), word_hit_callback,max_mode, <darts_ext> tokens)
        return tokens

        

    def __dealloc__(self):
        free_segment(self.sg)
        self.sg=NULL
        

        
    
def wordType(word):
    cdef char* ret=NULL
    cdef size=word_type(word.encode('utf-8','ignore'),&ret)
    try:
        return ret[:size].decode('utf-8')
    except:
        free(ret)

    

cdef bool token_hit_callback(const char* strs, const char* label, size_t s, size_t e, darts_ext user_data):
    tokens=<object>user_data
    txt=strs.decode('utf-8','ignore') if strs else ''
    tag=label.decode('utf-8','ignore') if label else ''
    tokens.append((txt,tag,s,e))

    
def wordSplit(strs):
    if not strs:return []
    tokens=[]
    word_split(strs.encode('utf-8','ignore'), token_hit_callback, <darts_ext> tokens)
    return tokens




