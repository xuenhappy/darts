from libc.stdlib cimport malloc, free
from libcpp cimport bool
import atexit
from typing import Iterator,Callable,List,Iterable,Sequence,Tuple
__all__=('normalize','wordDtype','Dregex','Token','Segment')

cdef extern from 'darts.h': 
    struct _segment:
        pass
    struct _dregex:
        pass
    struct _encoder:
        pass

    ctypedef _dregex* dregex
    ctypedef _segment* segment
    ctypedef _encoder* encoder 
    ctypedef void* ext_data
    
    ctypedef bool(*atom_iter)(const char**, size_t*, ext_data)
    ctypedef bool(*dregex_hit)(size_t, size_t, const char**, size_t, ext_data)
    ctypedef bool (*kv_iter)(const char**, size_t*, const char**, size_t*,size_t, size_t, ext_data)

    ctypedef void(*word_hit)(const char*, const char*, size_t, size_t, size_t, size_t ,ext_data)

    void init_darts_env()
    void destroy_darts_env()
    char* normalize_str(const char* strs, size_t lens, size_t* ret)
    const char* word_type(const char* word, size_t lens)

    int load_drgex(const char* path, dregex* regex)
    int compile_regex(const char* outpath, kv_iter kvs, ext_data user_data)
    void free_dregex(dregex regex)
    void parse(dregex regex, atom_iter atomlist, dregex_hit hit, ext_data user_data)


    int load_segment(const char* json_conf_file, segment* sg,const char* mode)
    void token_str(segment sg, const char* txt, size_t textlen, word_hit hit, bool max_mode, bool normal_before,ext_data user_data)
    void free_segment(segment sg)

    int load_encoder(const char* confdir, encoder* cdr)
    int free_encoder(encoder cdr)
    void encode_alist(encoder cdr, segment sg, const char* txt, size_t textlen, word_hit hit, ext_data user_data)
    
    

#init some thing
atexit.register(destroy_darts_env)
init_darts_env()


def normalize(content:str)->str:
    if content is None:
        return None
    py_byte_string= content.encode('utf-8','replace')
    cdef char* data = py_byte_string
    cdef int ret_size = 0
    cdef char* point=normalize_str(data,len(py_byte_string),&ret_size)
    if not point:
        return ""
    try:
        return point[:ret_size].decode('utf-8','replace')
    finally:
        free(point)


def wordDtype(word:str)->str:
    py_byte_string= word.encode('utf-8','ignore')
    cdef char* data= py_byte_string
    cdef const char* ret=word_type(data,len(py_byte_string))
    if not ret:
        return ""
    return ret[:].decode('utf-8','ignore')
    
   



#define some callback function
cdef bool atom_iter_func(const char** word, size_t* postion, ext_data user_data):
    str_iter:Iterator[tuple[str,int]]= (<tuple>user_data)[0]
    atom_info=next(str_iter)
    if atom_info is None:
        return False
    py_byte_string= atom_info[0].encode('utf-8','replace')
    word[0]=<char*>py_byte_string
    postion[0]=<size_t>atom_info[1]
    return True
    

cdef bool dregex_hit_callback(size_t s, size_t e, const char** labels, size_t labels_num, ext_data user_data):
    py_hit:Callable[[int,int,List[str]]] =(<tuple>user_data)[1]
    ret_labels:List[str] =[labels[i][:].decode("utf-8","ignore") for i in range(labels_num) if labels[i]]
    py_hit(s,e,ret_labels)
    return False

cdef bool kviter_func(const char** words, size_t* wlen, const char** labels, size_t* labelnum,size_t max_key_len,size_t max_labels_len, ext_data user_data):
    kviters:Iterator[Tuple[Sequence[str],Sequence[str]]]= (<tuple>user_data)[0]
    kv_info=next(kviters)
    if kv_info is None:
        return False
    key_list,label_list= kv_info
    wlen[0]=len(key_list)
    for i in range(min(len(key_list),max_key_len)):
        words[i]=key_list[i].encode("utf-8",'ignore')

    if label_list is None or len(label_list)<1:
        labelnum[0]= 0
        labels[0]=NULL
    else:
        labelnum[0]=len(label_list)
        for i in range(min(len(label_list),max_labels_len)):
            labels[i]=label_list[i].encode("utf-8",'ignore')
    
    return True
    


cdef class Dregex:
    cdef dregex reg= NULL 
    __slots__=()

    def __cinit__(self, path:str):
        assert path is not None,"path must be give"
        py_byte_string= path.encode("utf-8",'ignore')
        if load_drgex(py_byte_string, &self.reg):
            raise IOError("load %s regex file failed!"%path)

    def parse(self, atoms:Iterable[str],hit:Callable[[int,int,List[str]]]):
        cdef ext_data user_data =<ext_data>(atoms,hit)
        parse(self.reg,atom_iter_func,dregex_hit_callback, user_data)

    @staticmethod
    def compile(outpath:str,kv_pairs:Iterator[Tuple[Sequence[str],Sequence[str]]]):
        if outpath is None or kv_pairs is None:
            return 
        py_byte_string= outpath.encode("utf-8",'ignore')
        cdef const char* path=py_byte_string
        if compile_regex(path,kviter_func,<ext_data>(kv_pairs)):
            raise IOError("compile and write [%s] failed!"%outpath)
        

    def __dealloc__(self):
        free_dregex(self.reg)
        self.reg=NULL




        

class Token:
    __slots__=()
    def __init__(self):
        self.image:str= None
        self.str_start:int= -1
        self.stra_end:int= -1
        self.a_start:int=0
        self.a_end:int =0
        self.tags:str= None

cdef void segment_callback(const char* strs, const char* label, size_t ast, size_t aet, size_t wst, size_t wet,ext_data user_data):
    tokens:list= <list>user_data
    t=Token()
    if strs:
        t.word=strs[:].decode("utf-8",'ignore')
    if label:
        t.tags=label[:].decode("utf-8",'ignore')
    t.start_of_string=ast
    t.end_of_string=aet
    t.start_of_atoms=wst
    t.end_of_atoms=wet
    tokens.append(t)


cdef class Segment:
    cdef segment sg= NULL

    def __cinit__(self, path:str=None,mode:str=None):
        if path is not None:
            path="data/darts.conf.json"
        if mode is None:
            mode=""
        py_path= path.encode("utf-8",'ignore')
        py_mode= mode.encode("utf-8",'ignore')
        flag=load_segment(py_path, &self.sg,py_mode)
        if flag or (not self.sg):
            raise IOError("load %s segment conf json file mode[%s] failed!"%(path,mode))

    def tokenize(self,strs:str, max_mode:bool=False, normal_before:bool=False)->List[Token]:
        if not strs:
            return []
        tokens:List[Token] =[]
        py_string=strs.encode('utf-8','replace')
        cdef char* data=py_string
        token_str(self.sg, data,len(py_string), segment_callback, max_mode, normal_before, <ext_data> tokens)
        return tokens

        

    def __dealloc__(self):
        free_segment(self.sg)
        self.sg=NULL



class Aidx:
    __slots__=('code','kind_code','atom_postion')
    def __init__(self,code:int,kind_code:int,apos:int):
        self.code:int=code
        self.kind_code:int=kind_code
        self.atom_postion:int=apos

      

cdef class Encoder:
    cdef encoder cdr= NULL

    def __cinit__(self, path:str=None):
        assert path is not None
        py_path= path.encode("utf-8",'ignore')
        flag=load_encoder(py_path, &self.cdr)
        if flag or (not self.cdr):
            raise IOError("load %s encoder failed!"%path)

    def encode_atoms(self,strs:str,seg:Segment)->List[Aidx]:
        if not strs:
            return []
        tokens:List[Token] =[]
        py_string=strs.encode('utf-8','replace')
        cdef char* data=py_string
        
        return tokens


    def __dealloc__(self):
        free_encoder(self.cdr)
        self.cdr=NULL

        
