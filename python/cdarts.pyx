from libc.stdlib cimport malloc, free
cimport cdarts

#init some thing
cdarts.init_darts()
def destroy_darts():
    cdarts.destroy_darts()

def normalize(content):
    if content is None:
        return None
    content = content.encode('utf-8','replace')
    cdef char* point=NULL
    cdef size=cdarts.normalize_string(content,&point)
    try:
        pyStr=point[:size].decode('utf-8','replace')
    finally:
        free(point)


#define some callback function
cdef bool atom_iter_func(const char** word, size_t* postion, cdarts.darts_ext user_data):
    atom_info=next((<object>user_data)[0])
    if(atom_info is None):
        return False
    atom, idx=atom_info
    *word=atom.encode('utf-8','replace')
    *postion=idx
    return True

cdef bool dregex_hit (size_t s, size_t e, const char** labels, size_t labels_num, cdarst.darts_ext user_data):
    py_hit=(<object>user_data)[1]
    labels_py=[]
    if(labels and labels_num>0):
        labels_py=[labels[i].decode("utf-8","ignore") for i in range(labels_num) if labels[i]]
    py_hit(s,e,labels_py)
    return False


cdef class Dregex:
    cdef cdarts.dregex data

    def __cinit__(self, path):
        assert path is not None,"path must be give"
        self.data=NULL
        cdarts.load_drgex(path.encode("utf-8",'ignore'), &self.data)
        if not self.data:
            raise IOError("load %s regex file failed!"%path)

    def parse(self, atom_iter,hit):
        user_data =(atom_iter,hit)
        cdarts.parse(self.data,atom_iter_func,dregex_hit, <cdarts.darts_ext>user_data)
        

    def __dealloc__(self):
        cdarts.free_dregex(self.data)
        self.data=NULL
        

class Token:
    def __init__(self):
        self.word=None
        self.start_of_string=0
        self.end_of_string=0
        self.start_of_atoms=0
        self.end_of_atoms=0
        self.tags=None

cdef void word_hit(const char* strs, const char* label, size_t ast, size_t aet, size_t wst, size_t wet,darts_ext user_data):
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
    cdef cdarts.segment sg

    def __cinit__(self, path=None):
        if path is not None:
            path="data/darts.conf.json"
        self.sg=NULL
        cdarts.load_segment(path.encode("utf-8",'ignore'), &self.sg)
        if not self.sg:
            raise IOError("load %s segment conf json file failed!"%path)

    def tokenize(self,strs,max_mode=False):
        if not strs:
            return []
        tokens=[]
        cdarts.token_str(self.sg, strs.encode('utf-8','replace'), word_hit,max_mode, <cdarts.darts_ext> tokens)
        return tokens

        

    def __dealloc__(self):
        cdarts.free_segment(self.sg)
        self.sg=NULL
        

        
    
def wordType(word):
    cdef char* ret=NULL
    cdef size=cdarts.word_type(word,encode('utf-8','ignore'),ret)
    try:
        return ret[:size].decode('utf-8')
    except:
        free(ret)

    

cdef bool token_hit(const char* strs, const char* label, size_t s, size_t e, cdats.darts_ext user_data):
    tokens=<object>user_data
    txt=if strs:strs.decode('utf-8','ignore') else ''
    tag=if label:label.decode('utf-8','ignore') else ''
    user_data.append((txt,tag,s,e))

    
def wordSplit(strs):
    if not strs:return []
    tokens=[]
    cdarts.word_split(strs.encode('utf-8','ignore'), token_hit, <cdats.darts_ext> tokens)
    return tokens





