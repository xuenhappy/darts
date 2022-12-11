from libc.stdlib cimport malloc, free
from libcpp cimport bool
import atexit
from typing import Iterator,Callable,List,Iterable,Sequence,Tuple


cdef extern from 'darts.h':
    struct _dregex 
    struct _decider
    struct _atomlist
    struct _wordlist
    struct _segment
    struct _encoder

    ctypedef _dregex* dreg
    ctypedef _decider* decider
    ctypedef _atomlist* atomlist
    ctypedef _wordlist* wordlist
    ctypedef _segment* segment
    ctypedef _encoder* encoder

    
    void init_darts_env()
    void destroy_darts_env()

  
    char* normalize_str(const char* str, size_t len, size_t* ret)
    const char* chtype(const char* word)

    struct atom_:
        const char* image
        size_t st
        size_t et
        const char* char_type
        bool masked
    
    
    ctypedef bool (*walk_alist_hit)(void* , atom_* )
    atomlist asplit(const char* txt, size_t textlen, bool skip_space, bool normal_before)
    void free_alist(atomlist alist)
    void walk_alist(atomlist alist, walk_alist_hit hit, void* user_data)
    size_t alist_len(atomlist alist)

    
    dreg load_dregex(const char* path)
    void free_dregex(dreg regex)

    struct atomiter_ret:
        const char* word
        size_t len
        size_t postion
    
    ctypedef bool (*atomiter)(void*, atomiter_ret*)

    struct dhit_ret:
        size_t s
        size_t e
        const char** labels
        size_t labels_size
    
    ctypedef bool (*dhit)(void*, dhit_ret*)
    struct kviter_ret:
        char** key
        size_t keylen
        char** labels
        size_t label_nums

    ctypedef bool (*kviter)(void*, kviter_ret*)
    void parse(dreg regex, atomiter atomlist, dhit hit, void* user_data)
    int compile_regex(const char* outpath, kviter kvs, void* user_data)

    segment load_segment(const char* conffile, const char* mode, bool isdevel)
    void free_segment(segment sg)
    wordlist token_str(segment sg, atomlist alist, bool max_mode)
    void free_wordlist(wordlist wlist)

#init some thing
atexit.register(destroy_darts_env)
init_darts_env()


def normalize(content:str)->str:
    if content is None:
        return None
    py_byte_string= content.encode('utf-8','replace')
    cdef char* data = py_byte_string
    cdef size_t ret_size = 0
    cdef char* point=normalize_str(data,len(py_byte_string),&ret_size)
    if not point:
        return ""
    try:
        return point[:ret_size].decode('utf-8','replace')
    finally:
        free(point)


def charDtype(unichr:str)->str:
    py_byte_string= unichr.encode('utf-8','ignore')
    cdef char* data= py_byte_string
    cdef const char* ret=chtype(data)
    if not ret:
        return ""
    return ret[:].decode('utf-8','ignore')
    
   



#define some callback function
cdef bool atomiter_func(void* user_data,atomiter_ret *ret):
    str_iter:Iterator[tuple[str,int]]= (<tuple>user_data)[0]
    try:
        atom_info=next(str_iter)
        py_byte_string= atom_info[0].encode('utf-8','replace')
        ret.word=<char*>py_byte_string
        ret.len=len(atom_info[0])
        ret.postion=<size_t>atom_info[1]
    except StopIteration:
        return False
    return True
    
    

cdef bool dregex_hit_callback(void* user_data, dhit_ret* ret):
    py_hit:Callable[[int,int,List[str]]] =(<tuple>user_data)[1]
    ret_labels:List[str] =[]
    if ret.labels_size>0 and ret.labels:
        for i in range(ret.labels_size):
            if not ret.labels[i]:
                continue
            ret_labels.append(ret.labels[i][:].decode("utf-8","ignore"))
    py_hit(ret.s,ret.e,ret_labels)
    return False


ctypedef char* cstr
cdef void copy_str_data(cstr** rets,size_t* lens,strs:List[str]):
    if len(strs)>lens[0]:
        if rets[0]:free(rets[0])
        rets[0] = <char **>malloc(len(strs) * sizeof(cstr))

    lens[0]=len(strs)
    for i in range(len(strs)):
        py_byte_string= strs[i].encode("utf-8",'ignore')
        rets[0][i] = py_byte_string
    

cdef bool kviter_func(void* user_data, kviter_ret* ret):
    kviters:Iterator[Tuple[Sequence[str],Sequence[str]]]= (<tuple>user_data)[0]
    try:
        kv_info=next(kviters)
        key_list,label_list= kv_info
        copy_str_data(&ret.key,&ret.keylen,key_list)
        copy_str_data(&ret.labels,&ret.label_nums,label_list)
    except StopIteration:
        if ret.key:free(ret.key)
        if ret.labels:free(ret.labels)
        return False
    
    return True
    


cdef class Dregex:
    cdef dreg reg 
    __slots__=()

    def __cinit__(self, path:str):
        assert path is not None,"path must be give"
        py_byte_string= path.encode("utf-8",'ignore')
        self.reg=load_dregex(py_byte_string)
        if not self.reg:
            raise IOError("load %s regex file failed!"%path)

    def parse(self, atoms:Iterable[str],hit:Callable[[int,int,List[str]]]):
        py_user_data=(iter(atoms),hit)
        cdef void* user_data =<void*>py_user_data
        parse(self.reg, atomiter_func, dregex_hit_callback , user_data)

    @staticmethod
    def compile(outpath:str,kv_pairs:Iterator[Tuple[Sequence[str],Sequence[str]]]):
        if outpath is None or kv_pairs is None:
            return 
        py_byte_string= outpath.encode("utf-8",'ignore')
        cdef const char* path=py_byte_string
        pydata=(kv_pairs,)
        if compile_regex(path,kviter_func,<void*>pydata):
            raise IOError("compile and write [%s] failed!"%outpath)
        

    def __dealloc__(self):
        free_dregex(self.reg)
        self.reg=NULL




