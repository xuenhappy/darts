/**
 * @file dregex.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-12-25
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef SRC_UTILS_DREGEX_HPP_
#define SRC_UTILS_DREGEX_HPP_

#include <queue>
#include <vector>
#include <map>
namespace darts {


class Trie {
    private:
        std::vector<int64_t>  Check, Base, Fail, L;
        std::vector<std::vector<int64_t>* > V,OutPut;              
	    size_t MaxLen;
	    std::map<std::string,int> CodeMap;

        int64_t getcode(const std::string &word)const{
            auto it=this->CodeMap.find(word);
            if(it==this->CodeMap.end()){return this->CodeMap.size()+1;}
            return it->second; 
        }

        //trans
        int64_t transitionWithRoot(int64_t nodePos , int64_t c ) const {
            int64_t b = 0;
            if (nodePos <t.Base.size()) {
                b = t.Base[nodePos];
            }
            auto p = b + c + 1;
            auto x = 0;
            if (p < t.Check.size()) {
                x = t.Check[p];
            }
            if (b != x) {
                if (nodePos == 0) {
                    return 0;
                }
                return -1;
            }
            return p;
        }

        // get state transpose
        int64_t getstate(int64_t currentState,int64_t character) const {
            auto newCurrentState = this->transitionWithRoot(currentState, character);
            while (newCurrentState == -1) {
                currentState = this->Fail[currentState];
                newCurrentState = this->transitionWithRoot(currentState, character);
            }
            return newCurrentState;
        }

        //ParseText parse a text list hit ( [start,end),tagidx)
        void parse(text StringIter, std::function<bool(size_t, size_t, const std::vector<int64_t>*)> hit) {
            auto currentState=0, indexBufferPos=0;
            std::vector<int64_t> indexBufer;
            indexBufer.assign(this->MaxLen,0);
            text(func(seq *string, position int) bool {
                indexBufer[indexBufferPos%this->MaxLen] = position
                indexBufferPos++
                currentState = this->getstate(currentState, this->getcode(seq))
                auto hitArray = t->OutPut[currentState]
                for ( auto h :hitArray ){
                    auto preIndex = (indexBufferPos - t.L[h]) % t.MaxLen
                    if(hit(indexBufer[preIndex], position+1, t.V[h])){
                        return;
                    }
                }
                
            })

        }





}



// get string code







//State is DFA mechine state
type State struct {
	depth   int          // the string length
	failure *State       // match failed use
	emits   []int        // emits
	success *treemap.Map //go map
	index   int          //index of the struct
}

func newState(depth int) *State {
	S := new(State)
	S.depth = depth
	S.success = treemap.NewWithIntComparator()
	return S
}

//insert sort a sort to keep it order
func insertSorted(s []int, e int) []int {
	i := sort.Search(len(s), func(i int) bool { return (s)[i] <= e })
	if i == len(s) {
		s = append(s, e)
		return s
	}
	if s[i] == e {
		return s
	}
	s = append(s, 0)
	copy(s[i+1:], s[i:])
	s[i] = e
	return s
}

//add emit
func (s *State) addEmit(keyword int) {
	if keyword != -100000 {
		s.emits = insertSorted(s.emits, keyword)
	}
}

//get l code
func (s *State) getMaxValueIDgo() int {
	if len(s.emits) < 1 {
		return -100000
	}
	return s.emits[0]
}

func (s *State) isAcceptable() bool {
	return s.depth > 0 && len(s.emits) > 0
}

func (s *State) setFailure(failState *State, fail []int) {
	s.failure = failState
	fail[s.index] = failState.index
}

func (s *State) nextState(character int, ignoreRootState bool) *State {
	nextState, _ := s.success.Get(character)

	if (!ignoreRootState) && (nextState == nil) && (s.depth == 0) {
		nextState = s
	}
	if nextState == nil {
		return nil
	}
	return nextState.(*State)
}

func (s *State) addState(character int) *State {
	nextS := s.nextState(character, true)
	if nextS == nil {
		nextS = newState(s.depth + 1)
		s.success.Put(character, nextS)
	}
	return nextS
}

//Builder is inner useed
type Builder struct {
	rootState *State
	trie      *Trie
	/**
	* whether the position has been used
	 */
	used []bool
	/**
	* the allocSize of the dynamic array
	 */
	allocSize int
	/**
	* the next position to check unused memory
	 */
	nextCheckPos int
	/**
	* the size of the key-pair sets
	 */
	size int
}

//zip no use data
func (b *Builder) zipWeight() {
	msize := b.size
	newBase := make([]int, msize)
	copy(newBase[:b.size], b.trie.Base[:b.size])
	b.trie.Base = newBase

	newCheck := make([]int, msize, msize)
	copy(newCheck[:b.size], b.trie.Check[:b.size])
	b.trie.Check = newCheck
}

func (b *Builder) constructFailureStates() {
	b.trie.Fail = make([]int, b.size+1)
	b.trie.OutPut = make([][]int, b.size+1)
	queue := list.New()

	cpy := func(ori []int) []int {
		dest := make([]int, len(ori))
		copy(dest, ori)
		return dest
	}

	for _, state := range b.rootState.success.Values() {
		depthOneState := state.(*State)
		depthOneState.setFailure(b.rootState, b.trie.Fail)
		queue.PushBack(state)
		if len(depthOneState.emits) > 0 {
			b.trie.OutPut[depthOneState.index] = cpy(depthOneState.emits)
		}

	}
	for queue.Len() > 0 {
		currentState := queue.Remove(queue.Front()).(*State)
		for _, key := range currentState.success.Keys() {
			transition := key.(int)
			targetState := currentState.nextState(transition, false)
			queue.PushBack(targetState)
			traceFailureState := currentState.failure
			for traceFailureState.nextState(transition, false) == nil {
				traceFailureState = traceFailureState.failure
			}
			newFailureState := traceFailureState.nextState(transition, false)
			targetState.setFailure(newFailureState, b.trie.Fail)
			for _, e := range newFailureState.emits {
				targetState.addEmit(e)
			}
			b.trie.OutPut[targetState.index] = cpy(targetState.emits)
		}
	}
}

func (b *Builder) addAllKeyword(kvs StrLabelIter) int {
	maxCode, index, t := 0, -1, b.trie
	kvs(func(k StringIter, v []int) bool {
		index++
		lens := 0
		currentState := b.rootState
		k(func(s *string, _ int) bool {
			lens++
			code := b.trie.getcode(s)
			t.CodeMap[*s] = code
			currentState = currentState.addState(code)
			return false
		})
		currentState.addEmit(index)
		t.L = append(t.L, lens)
		if lens > t.MaxLen {
			t.MaxLen = lens
		}
		maxCode += lens
		t.V = append(t.V, v)

		return false
	})
	t.MaxLen++
	return int(float32(maxCode+len(t.CodeMap))/1.5) + 1
}

//resize data
func (b *Builder) resize(newSize int) {
	base2 := make([]int, newSize)
	check2 := make([]int, newSize)
	used2 := make([]bool, newSize)
	if b.allocSize > 0 {
		copy(base2[:int(b.allocSize)], b.trie.Base[:int(b.allocSize)])
		copy(check2[:int(b.allocSize)], b.trie.Check[:int(b.allocSize)])
		copy(used2[:int(b.allocSize)], b.used[:int(b.allocSize)])
	}
	b.trie.Base = base2
	b.trie.Check = check2
	b.used = used2
	b.allocSize = newSize
}

//Pair si tmp used
type Pair struct {
	K interface{}
	V interface{}
}

func fetch(parent *State) []Pair {
	siblings := make([]Pair, 0, parent.success.Size()+1)
	if parent.isAcceptable() {
		fakeNode := newState(-(parent.depth + 1))
		fakeNode.addEmit(parent.getMaxValueIDgo())
		siblings = append(siblings, Pair{0, fakeNode})
	}
	it := parent.success.Iterator()
	for it.Next() {
		siblings = append(siblings, Pair{it.Key().(int) + 1, it.Value()})
	}
	return siblings
}

func (b *Builder) insert(queue *list.List) {
	tCurrent := queue.Remove(queue.Front()).(Pair)
	value, siblings := tCurrent.K.(int), tCurrent.V.([]Pair)

	begin, nonZeroNum := 0, 0
	first := true
	pos := b.nextCheckPos - 1
	if pos < siblings[0].K.(int) {
		pos = siblings[0].K.(int)
	}
	if b.allocSize <= pos {
		b.resize(pos + 1)
	}
	t := b.trie
	for {
		pos++
		if b.allocSize <= pos {
			b.resize(pos + 1)
		}
		if t.Check[pos] != 0 {
			nonZeroNum++
			continue
		} else if first {
			b.nextCheckPos = pos
			first = false
		}

		begin = pos - siblings[0].K.(int)
		if b.allocSize <= (begin + siblings[len(siblings)-1].K.(int)) {
			b.resize(begin + siblings[len(siblings)-1].K.(int) + 100)
		}
		if b.used[begin] {
			continue
		}
		allIszero := true
		for i := 0; i < len(siblings); i++ {
			if t.Check[begin+siblings[i].K.(int)] != 0 {
				allIszero = false
				break
			}
		}
		if allIszero {
			break
		}
	}

	if float32(nonZeroNum)/float32(pos-b.nextCheckPos+1) >= 0.95 {
		b.nextCheckPos = pos
	}
	b.used[begin] = true

	if b.size < begin+siblings[len(siblings)-1].K.(int)+1 {
		b.size = begin + siblings[len(siblings)-1].K.(int) + 1
	}
	for i := 0; i < len(siblings); i++ {
		t.Check[begin+siblings[i].K.(int)] = begin
	}
	for i := 0; i < len(siblings); i++ {
		kv := siblings[i]
		newSiblings := fetch(kv.V.(*State))
		if len(newSiblings) < 1 {
			t.Base[begin+kv.K.(int)] = -(kv.V.(*State).getMaxValueIDgo() + 1)
		} else {
			queue.PushBack(Pair{begin + kv.K.(int), newSiblings})
		}
		kv.V.(*State).index = begin + kv.K.(int)
	}
	if value >= 0 {
		t.Base[value] = begin
	}
}

func (b *Builder) build(kvs StrLabelIter) {
	maxCode := b.addAllKeyword(kvs)
	//build double array tire base on tire
	b.resize(maxCode + 10)
	b.trie.Base[0] = 1
	siblings := fetch(b.rootState)
	if len(siblings) > 0 {
		queue := list.New()
		queue.PushBack(Pair{-1, siblings})
		for queue.Len() > 0 {
			b.insert(queue)
		}
	}
	//build failure table and merge output table
	b.constructFailureStates()
	b.rootState = nil
	b.zipWeight()
}

//CompileTrie the data
func CompileTrie(kvs StrLabelIter) *Trie {
	var builder Builder
	builder.rootState = newState(0)
	builder.trie = newTrie()
	builder.build(kvs)
	//clear mem
	result := builder.trie
	builder.trie = nil
	return result
}

}
#endif  // SRC_UTILS_DREGEX_HPP_
