from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
import atexit
from typing import Iterator,Callable,List,Iterable,Tuple


cdef extern from 'darts.h':
    ctypedef struct _dregex 
    ctypedef struct _decider
    ctypedef struct _atomlist
    ctypedef struct _wordlist
    ctypedef struct _segment
    ctypedef struct _encoder

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

    ctypedef struct atom_:
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

    ctypedef struct atomiter_ret:
        const char* word
        size_t len
        size_t postion
    
    ctypedef bool (*atomiter)(void*, atomiter_ret*)

    ctypedef struct dhit_ret:
        size_t s
        size_t e
        const char** labels
        size_t labels_size
    
    ctypedef bool (*dhit)(void*, dhit_ret*)
    ctypedef struct kviter_ret:
        void* key_cache
        void* label_cache
       

    ctypedef bool (*kviter)(void*, kviter_ret*)
    void parse(dreg regex, atomiter atomlist, dhit hit, void* user_data)
    int compile_regex(const char* outpath, kviter kvs, void* user_data)


    ctypedef struct wordlist_ret:
        size_t atom_s, atom_e
        size_t code_s, code_e
        const char** labels
        size_t label_nums
        const char* image
    

    ctypedef bool (*walk_wlist_hit)(void* , wordlist_ret* )
    void walk_wlist(wordlist wlist, walk_wlist_hit hit, void* user_data)
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
        atom_info=<tuple>next(str_iter)
        chrs=<str>atom_info[0]
        py_byte_string= chrs.encode('utf-8','replace')
        ret.word=<char*>py_byte_string
        ret.len=len(py_byte_string)
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


ctypedef const char* cstr
ctypedef vector[cstr]* ctsr_list
cdef void copy_str_data(ctsr_list clist,list strs):
    for i in range(len(strs)):
        item=<str>strs[i]
        py_byte_string= item.encode("utf-8",'ignore')
        clist.push_back(py_byte_string)
       
    
cdef bool kviter_func(void* user_data, kviter_ret* ret):
    key_cache=<ctsr_list>(ret.key_cache)
    label_cache=<ctsr_list>(ret.label_cache)

    kviters:Iterator[Tuple[List[str],List[str]]]= (<tuple>user_data)[0]
    try:
        kv_info=<tuple>next(kviters)
        copy_str_data(key_cache,<list>(kv_info[0]))
        copy_str_data(label_cache,<list>(kv_info[1]))
    except StopIteration:
        return False

    return True
    


cdef class Dregex:
    cdef dreg reg 
    __slots__=()

    def __cinit__(self, path:str):
        assert path is not None,"path must be give"
        py_byte_string= path.encode("utf-8",'ignore')
        self.reg=load_dregex(py_byte_string)
        if self.reg==NULL:
            raise IOError("load %s regex file failed!"%path)

    def parse(self, atoms:Iterable[str],hit:Callable[[int,int,List[str]]]):
        py_user_data=(iter(atoms),hit)
        cdef void* user_data =<void*>py_user_data
        parse(self.reg, atomiter_func, dregex_hit_callback , user_data)

    @staticmethod
    def compile(outpath:str,kv_pairs:Iterator[Tuple[List[str],List[str]]]):
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


cdef class PyAtom:
    cdef str image
    cdef size_t st
    cdef size_t et
    cdef char_type
    cdef bool masked

    def __repr__(self) -> str:
        if self.masked:
            return "[MASK]"
        return self.image

    




cdef bool alist_hit_func(void* user_data, atom_* a):
    ret=<list>user_data
    atm=PyAtom();
    atm.image=a.image[:].decode("utf-8","ignore")
    atm.st=a.st
    atm.et=a.et
    atm.masked=a.masked
    atm.char_type=a.char_type[:].decode("utf-8","ignore")
    ret.append(atm)
    return False

cdef class PyAtomList:
    cdef atomlist alist 
    __slots__=()

    def __cinit__(self, str text,bool skip_space=True, bool normal_before=True):
        assert text is not None
        py_byte_string= text.encode("utf-8",'ignore')
        self.alist=asplit(py_byte_string,len(py_byte_string),skip_space,normal_before)

    def tolist(self):
        user_data=[]
        walk_alist(self.alist,alist_hit_func, <void*> user_data)
        return user_data

    def size(self)->size_t:
        return alist_len(self.alist)

    def __dealloc__(self):
        free_alist(self.alist)
        self.alist=NULL



cdef class PyWord:
    cdef str image
    cdef size_t atom_s,atom_e
    cdef size_t code_s,code_e
    cdef set labels

    def __repr__(self) -> str:
        return self.image
   
cdef bool wlist_hit_func(void* user_data, wordlist_ret* w):
    ret=<list>user_data
    word=PyWord();
    word.image=w.image[:].decode("utf-8","ignore") if w.image!=NULL else ""
    word.atom_s=w.atom_s
    word.atom_e=w.atom_e
    word.code_s=w.code_s
    word.code_e=w.code_e
    if w.label_nums>0 and w.labels!=NULL:
        for i in range(w.label_nums):
            label=w.labels[i]
            if label==NULL:
                continue
            pylabel=label[:].decode("utf-8","ignore")
            word.labels.add(pylabel)
    ret.append(word)
    return False

cdef class DSegment:
    cdef segment segt 

    def __cinit__(self,conffile:str,mode:str,isdev:bool):
        if conffile is None or len(conffile)<2:
            raise IOError("conf file must given!")
        py_bytes=conffile.encode("utf-8","ignore")
        cdef const char* confpath=py_bytes
        cdef const char* modstr=NULL
        if mode:
            py_bytes=mode.encode("utf-8","ignore")
            modstr=py_bytes
        self.segt=load_segment(confpath, modstr, isdev)
        if self.segt==NULL:
            raise IOError(f"load {conffile} segment failed!")

    def cut(self,strs:str,max_mode:bool =False):
        if not strs or len(strs)<3:
            return strs
        alist=PyAtomList(strs)
        cdef wordlist wlist=token_str(self.segt,alist.alist,max_mode)
        try:
            ret_list=[]
            walk_wlist(wlist, wlist_hit_func, <void*> ret_list)
            return ret_list
        finally:
            free_wordlist(wlist)



    def __dealloc__(self):
        free_segment(self.segt)
        self.segt=NULL