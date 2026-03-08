'''
in one chapter , i will to implement tokenizer basic
it means will text(string) convert easy process data seriese(token) use for computer
'''


def get_pairs(ids):
    counts = {}
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts

def merge(ids,pair,index):
    """
    this function will iterate ids  and every time 
    it sees a instanmce a pair ,it will take that pair 
    and instead put index,then it will return list
    list = [1,2,3,4,1,2]
    merge(list,(1,2),257)
    list = [257,3,4,257]
    """       
    new_ids = []
    i  = 0
    while i < len(ids):
        if i< len(ids)-1 and (ids[i] , ids[i+1] ) == pair:
            new_ids.append(index)
            i+=2
        else:
            new_ids.append(ids[i])
            i +=1
    return new_ids

class BasicTokenizer:
    def __init__(self,vocab_size):
        '''
        vocab_size : you will tokenizer reconize different signal or word
        vocabulary : tokenizer inital byte list example 65 - > b'A' , worl anything convery byte (chinese , english ,emoji)
        bytes(256)

        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00
        \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        '''
        self.vocab_size=vocab_size
        self.vocabulary= { i :bytes([i]) for i in range(256) }
        self.merges = {}
    def train(self,text,verbose=False):
        # encode the text
        # iteratr over text, self.vocab_size - 256 times
        # rach time all the pairs in a dictory
        # chooose the pair with the highest frequency
        # merge that pair as a new token
        # add that token to the vocab
        # { 256: byte_string }
        # add to self.merges = {byte_string : 256}

        assert self.vocab_size > 256
        number_merges = self.vocab_size - 256
        byte_strings = text.encode("utf-8")
        ids = list(byte_strings)
        length_inital = len(ids)
        # print(ids)
        for i in range(number_merges):
            pairs = get_pairs(ids)
            pair = max(pairs, key=pairs.get)
            index = 256 + i
            ids = merge(ids,pair,index)
            self.merges[pair] = index
            self.vocabulary[index] =self.vocabulary[pair[0]]+ self.vocabulary[pair[1]]
            
        if verbose:
            length_final = len(ids)
            compression = length_inital / len(ids)
            print(length_inital,length_final)
            print(compression)
            # 104 89
            # 1.1685393258426966
        #print(ids)
        
        
        #print(sorted([(v,k) for k,v in pairs.items()],reverse=True)[:10])
        #print(sorted(pairs.items(),reverse=True,key=lambda k : pairs[k])[:10])
    def encode(self,text):
        """
        self.merges is important here
        we get text ,and then eewconvert text to byte string ,then to integres
        *** we merge the pairs in the order they were merged at training ***
        and the we iterate the text until all pairs of merges that are
        possible under the tains tokenizor have completed
        """
        byte_strings = text.encode("utf-8")
        ids = list(byte_strings)
        while len(ids) > 1:
            pairs= get_pairs(ids)
            '''
            pair is a dictory of tuples which tell us the frequency od each pair in the text to be the encoded
            we dont careabout the frequency here, because we are not training
             we want to find the pair with the minimmunm index ,THAT WAS MERGED
             key will take the key of pairs we input that pair into aginsts self.merges
            '''
            pair = min(pairs,key= lambda p : self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids,pair,self.merges[pair])
        return ids

    def decode(self,ids):
        """
        decode get ids 
        1 . convert the ids to their strings 
        2. convery the byte strings to strings via the vocabulary
        3. then return the decoded_text
        [239,256]
        [b'xa',b'sa']
        b'xasa'
        output
        """
        byte_strings = b''.join(self.vocabulary[i] for i in ids)
        decoded_text = byte_strings.decode("utf-8")
        #print(decoded_text)
        return decoded_text
        # print("hello".encode("utf-8"))
        #print(byte_strings)

tokenizor = BasicTokenizer(266)
text =  """
Hello World! 
这是一个基础分词器测试。
Python 编程 😊 🚀
Numeric: 12345, Symbol: @#&
"""
tokenizor.train(text,False)

# print(tokenizor.merges)
print(tokenizor.encode(" are hello"))
print(tokenizor.decode(tokenizor.encode("are hello")))
print(list[int]("are hello".encode("utf-8")))