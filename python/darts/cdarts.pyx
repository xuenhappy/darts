from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair
import atexit
from typing import Iterator,Callable,List,Iterable,Tuple


cdef extern from 'darts.h':
    ctypedef struct _dregex 
    ctypedef struct _atomlist
    ctypedef struct _wordlist
    ctypedef struct _segment
    ctypedef struct _alist_encoder
    ctypedef struct _wtype_encoder

    ctypedef _dregex* dreg
    ctypedef _atomlist* atomlist
    ctypedef _wordlist* wordlist
    ctypedef _segment* segment
    ctypedef _alist_encoder* alist_encoder
    ctypedef _wtype_encoder* wtype_encoder


    
    void init_darts_env()
    void destroy_darts_env()

  
    void normalize_str(const char* str, size_t len, void* cpp_string_cache) nogil
    const char* chtype(const char* word)

    ctypedef struct atom_buffer:
        const char* image
        size_t st
        size_t et
        const char* char_type
        bool masked    
    
    ctypedef bool (*walk_alist_hit)(void* , atom_buffer* )
    atomlist asplit(const char* txt, size_t textlen, bool skip_space, bool normal_before) nogil
    void free_alist(atomlist alist)
    void walk_alist(atomlist alist, walk_alist_hit hit, void* user_data)
    int get_npos_atom(atomlist alist, size_t idx, atom_buffer* buffer)
    size_t alist_len(atomlist alist)

    
    dreg load_dregex(const char* path)
    void free_dregex(dreg regex)

    ctypedef struct atomiter_buffer:
        const char* word
        size_t len
        size_t postion
    
    ctypedef bool (*atomiter)(void*, atomiter_buffer*)

    ctypedef struct dhit_buffer:
        size_t s
        size_t e
        const char** labels
        size_t labels_size
    
    ctypedef bool (*dhit)(void*, dhit_buffer*)
    ctypedef struct kviter_buffer:
        void* key_cache
        void* label_cache
       

    ctypedef bool (*kviter)(void*, kviter_buffer*)
    void parse(dreg regex, atomiter atomlist, dhit hit, void* user_data)
    int compile_regex(const char* outpath, kviter kvs, void* user_data)

    int build_biggram_dict(const char* single_freq_dict, const char* union_freq_dict, const char* outdir)


    ctypedef struct word_buffer:
        void* label_cache
        size_t atom_s, atom_e
        const char** labels
        size_t label_nums
        const char* image
    

    ctypedef bool (*walk_wlist_hit)(void* , word_buffer* )
    void walk_wlist(wordlist wlist, walk_wlist_hit hit, void* user_data)
    segment load_segment(const char* conffile, const char* mode, bool isdevel)
    void free_segment(segment sg)

    size_t wlist_len(wordlist wlist)
    int get_npos_word(wordlist wlist, size_t index, word_buffer* buffer)
    wordlist token_str(segment sg, atomlist alist, bool max_mode) nogil
    void free_wordlist(wordlist wlist)

    wtype_encoder get_wtype_encoder(void* map_param,const char* type_cls_name)
    void encode_wlist_type(wtype_encoder encoder, wordlist wlist, void* int_vector_buf) nogil
    void free_wtype_encoder(wtype_encoder encoder)
    size_t max_wtype_nums(wtype_encoder encoder)
    const char* decode_wtype(wtype_encoder encoder,int wtype)

    alist_encoder get_alist_encoder(void* map_param,const char* type_cls_name)
    void encode_alist(alist_encoder encoder, atomlist alist, void* int_pair_vector_buf) nogil
    void free_alist_encoder(alist_encoder encoder)
    size_t max_acode_nums(alist_encoder encoder) 
    const char* decode_atype(alist_encoder encoder,int atype)

ctypedef const char* cstr
ctypedef vector[cstr]* ctsr_list
#init some thing
atexit.register(destroy_darts_env)
init_darts_env()



def normalize(str content not None)->str:
    py_byte_string= content.encode('utf-8','replace')
    cdef const char* data = py_byte_string
    cdef size_t byteslen=len(py_byte_string)
    cdef string cache
    with nogil:
        normalize_str(data,byteslen,&cache)
    return cache.c_str().decode('utf-8','replace')

def charDtype(str unichr not None)->str:
    py_bytes=unichr.encode('utf-8','ignore')
    cdef const char* ret=chtype(py_bytes)
    if not ret:
        return ""
    return ret[:].decode('utf-8','ignore')
    
def build_gramdict_fromfile(str single_freq_dict not None, str union_freq_dict not None, str outdir not None):
    single_pbytes=single_freq_dict.encode('utf-8','ignore')
    union_pbytes=union_freq_dict.encode('utf-8','ignore')
    outdir_pbytes=outdir.encode('utf-8','ignore')
    if build_biggram_dict(single_pbytes,union_pbytes,outdir_pbytes):
        raise IOError("build biggram dict failed!")




    
cdef class Dregex:
    cdef dreg reg 

    @staticmethod
    cdef inline bool atomiter_func(void* user_data,atomiter_buffer *ret):
        tuple_data=<tuple>user_data
        cdef list bytes_cache=<list>(tuple_data[1])
        try:
            atom_info=<tuple>next(tuple_data[0])
            py_byte_string=(<str>atom_info[1]).encode("utf-8",'ignore')
            bytes_cache[0]=py_byte_string #add to cache 
            ret.word=py_byte_string
            ret.len=len(py_byte_string)
            ret.postion=<size_t>atom_info[0]
        except StopIteration:
            return False
        return True
    
    
    @staticmethod
    cdef inline bool dregex_hit_callback(void* user_data, dhit_buffer* buf):
        py_hit=(<tuple>user_data)[2]
        ret_labels=[]
        if buf.labels_size>0 and buf.labels:
            for i in range(buf.labels_size):
                if not buf.labels[i]:
                    continue
                ret_labels.append(buf.labels[i][:].decode("utf-8","ignore"))
        py_hit(<int>buf.s,<int>buf.e,ret_labels)
        return False


    @staticmethod
    cdef inline bool kviter_func(void* user_data, kviter_buffer* buf):
        tuple_data=<tuple>user_data
        cdef list kcache=<list>(tuple_data[1])
        cdef list vcache=<list>(tuple_data[2])
        cdef ctsr_list key_cache=<ctsr_list>(buf.key_cache)
        cdef ctsr_list label_cache=<ctsr_list>(buf.label_cache)
        
        try:
            kv_info=<tuple>next(tuple_data[0])
            key_strs=<list>(kv_info[0])
            label_strs=<list>(kv_info[1])
            kcache[:]=iter((<str>l).encode("utf-8",'ignore') for l in key_strs)
            vcache[:]=iter((<str>l).encode("utf-8",'ignore') for l in label_strs)
            for iterm in kcache:
                key_cache.push_back(<cstr>iterm)
            for iterm in vcache:
                label_cache.push_back(<cstr>iterm)
        except StopIteration:
            return False

        return True
   
    def __cinit__(self, str path not None):
        py_byte_string= path.encode("utf-8",'ignore')
        self.reg=load_dregex(py_byte_string)
        if self.reg==NULL:
            raise IOError("load %s regex file failed!"%path)

    def parse(self, atoms:Iterable[str],hit:Callable[[int,int,List[str]]]):
        py_user_data=(enumerate(atoms),[b""],hit)
        cdef void* user_data =<void*>py_user_data
        parse(self.reg, Dregex.atomiter_func, Dregex.dregex_hit_callback , user_data)

    @staticmethod
    def compile(str outpath not None,kv_pairs:Iterator[Tuple[List[str],List[str]]]):
        if kv_pairs is None:
            return 
        path=outpath.encode("utf-8",'ignore')
        pydata=(kv_pairs,[],[])
        if compile_regex(path,Dregex.kviter_func,<void*>pydata):
            raise IOError("compile and write [%s] failed!"%outpath)
        

    def __dealloc__(self):
        free_dregex(self.reg)
        self.reg=NULL


cdef class PyAtom:
    cdef readonly str image
    cdef readonly size_t st
    cdef readonly size_t et
    cdef readonly str chtype
    cdef public bool masked

    def __cinit__(self,image:str,size_t st,size_t et):
        self.image=image
        self.st=st
        self.et=et

    def __repr__(self) -> str:
        if self.masked:
            return "[MASK]"
        return self.image

    



cdef class PyAtomList:
    cdef atomlist alist

    @staticmethod
    cdef inline bool alist_hit_func(void* user_data, atom_buffer* buf):
        cdef list ret=<list>user_data
        cdef str py_str=buf.image[:].decode("utf-8","ignore")
        atm=PyAtom(py_str,buf.st,buf.et);
        atm.chtype=buf.char_type[:].decode("utf-8","ignore")
        atm.masked=buf.masked
        ret.append(atm)
        return False
   
    def __cinit__(self, str text not None,bool skip_space=True, bool normal_before=True):
        py_byte_string= text.encode("utf-8",'ignore')
        cdef const char* txt=py_byte_string
        cdef size_t byteslen=len(py_byte_string)
        with nogil:
            self.alist=asplit(txt,byteslen,skip_space,normal_before)
        if self.alist==NULL:
            raise MemoryError("alloc memort error!")

    def tolist(self)->List[PyAtom]:
        user_data=[]
        walk_alist(self.alist,PyAtomList.alist_hit_func, <void*> user_data)
        return user_data

    def __getiterm__(self,idx:size_t)->PyAtom:
        cdef atom_buffer buf
        if get_npos_atom(self.alist,idx,&buf):
            raise IndexError("index err %d"%idx)
        py_str=buf.image[:].decode("utf-8","ignore")
        atm=PyAtom(py_str,buf.st,buf.et);
        atm.masked=buf.masked
        atm.char_type=""
        if buf.char_type!=NULL:
            atm.char_type=buf.char_type[:].decode("utf-8","ignore")
        return atm

    def __len__(self)->int:
        return alist_len(self.alist)

    def __dealloc__(self):
        free_alist(self.alist)
        self.alist=NULL



cdef class PyWord:
    cdef readonly str image
    cdef readonly size_t atom_s,atom_e
    cdef readonly set labels

    def __cinit__(self,image:str,size_t a_s,size_t a_e):
        self.image=image
        self.atom_s=a_s
        self.atom_e=a_e

    def __repr__(self) -> str:
        return "%s[%d,%d]"%(self.image,self.atom_s,self.atom_e)
   



cdef class PyWordList:
    cdef wordlist wlist

    def __cinit__(self):
        self.wlist=NULL

    @staticmethod
    cdef inline bool wlist_hit_func(void* user_data, word_buffer* buf):
        cdef list ret=<list>user_data
        cdef str py_str=buf.image[:].decode("utf-8","ignore") if buf.image!=NULL else ""
        word=PyWord(py_str,buf.atom_s,buf.atom_e);
        if buf.label_nums>0 and buf.labels!=NULL:
            for i in range(buf.label_nums):
                label=buf.labels[i]
                if label==NULL:
                    continue
                pylabel=label[:].decode("utf-8","ignore")
                word.labels.add(pylabel)
        ret.append(word)
        return False

    def tolist(self)->List[PyWord]:
        if self.wlist==NULL: return []
        ret_list=[]
        walk_wlist(self.wlist, PyWordList.wlist_hit_func, <void*> ret_list)
        return ret_list

    def __getiterm__(self,index:size_t)->PyWord:
        cdef word_buffer buf
        cdef ctsr_list ptrs
        buf.label_cache=<void *>(&ptrs);
        if get_npos_word(self.wlist,index,&buf):
            raise IndexError("index err %d"%index)
        py_str=buf.image[:].decode("utf-8","ignore") if buf.image!=NULL else ""
        word=PyWord(py_str,buf.atom_s,buf.atom_e);
        if buf.label_nums>0 and buf.labels!=NULL:
            for i in range(buf.label_nums):
                label=buf.labels[i]
                if label==NULL:
                    continue
                pylabel=label[:].decode("utf-8","ignore")
                word.labels.add(pylabel)
        return word
        
    def __len__(self) -> int:
        return wlist_len(self.wlist)

    def __dealloc__(self):
        free_wordlist(self.wlist)
        self.wlist=NULL


cdef class DSegment:
    cdef segment segt 

    def __cinit__(self,str conffile,str mode,bool isdev=False):
        if conffile is None:
            conffile="data/conf.json"
        confpath=conffile.encode("utf-8","ignore")
        cdef const char* modstr=NULL
        if mode:
            mode_bytes=mode.encode("utf-8","ignore")
            modstr=mode_bytes
        self.segt=load_segment(confpath, modstr, isdev)
        if self.segt==NULL:
            raise IOError(f"load {conffile} segment failed!")

    def cut(self,strs:str,max_mode:bool=False):
        if not strs or len(strs)<1:
            return (None,None)
        alist=PyAtomList(strs)
        wlist=PyWordList()
        with nogil:
            wlist.wlist=token_str(self.segt,alist.alist,max_mode)
        return (alist,wlist)
       

    def __dealloc__(self):
        free_segment(self.segt)
        self.segt=NULL



cdef class AtomCodec:
    cdef alist_encoder encoder

    def __cinit__(self, dict str_params not None,str cls_name=None):
        if cls_name is None:
            cls_name="WordPice"
        pparams={k.encode():v.encode() for k,v in str_params.items()}
        cdef map[string,string] param=pparams
        py_bytes=cls_name.encode("utf-8")
        self.encoder=get_alist_encoder(&param,py_bytes)
        if self.encoder==NULL:
            raise IOError("init atom codex failed from param",str_params)

    def label_nums(self):
        return max_acode_nums(self.encoder)

    def decode(self,int label):
        cdef const char* str_label=decode_atype(self.encoder,label)
        if str_label!=NULL:
            return str_label[:].decode("utf-8")
        return ""

    def encode(self,PyAtomList alist not None):
        cdef vector[pair[int,int]] buf
        with nogil:
            encode_alist(self.encoder,alist.alist,&buf)
        return buf


    def __dealloc__(self):
        free_alist_encoder(self.encoder)
        self.encoder=NULL



cdef class WordCodec:
    cdef wtype_encoder encoder

    def __cinit__(self,dict str_params not None,str cls_name not None):
        pparams={k.encode():v.encode() for k,v in str_params.items()}
        cdef map[string,string] param=pparams
        py_bytes=cls_name.encode("utf-8")
        self.encoder=get_wtype_encoder(&param,py_bytes)
        if self.encoder==NULL:
            raise IOError("init word codex failed from param",str_params)

    def label_nums(self):
        return max_wtype_nums(self.encoder)

    def decode(self,int label):
        cdef const char* str_label=decode_wtype(self.encoder,label)
        if str_label!=NULL:
            return str_label[:].decode("utf-8")
        return ""


    def encode(self,PyWordList wlist not None):
        cdef vector[int] buf
        with nogil:
            encode_wlist_type(self.encoder,wlist.wlist,&buf)
        return buf



    def __dealloc__(self):
        free_wtype_encoder(self.encoder)
        self.encoder=NULL
