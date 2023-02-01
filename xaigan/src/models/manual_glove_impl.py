import numpy as np
import torch, os, json,  random
import logging
import bcolz
import pickle5 as pickle
from transformers import RobertaModel, RobertaTokenizer

class GloVeEmbbeding(torch.nn.Module):
    def __init__(self,dataset_name,glove_path, GRU_hidden_size, GRU_num_layers=1):
        super(self).__init__()
        self.text_emb_sz = 50 # as Glove embedding size is 50
        self.voab_sz=0 #initalization, will be updated
        self.embedding = self.create_emb_layer_for_Target_Vocab(dataset_name,glove_path)
        self.GRU_hidden_size = GRU_hidden_size
        self.GRU_num_layers = GRU_num_layers
        self.gru = torch.nn.GRU(self.text_emb_sz, GRU_hidden_size, GRU_num_layers, batch_first=True)
        #if batch_first=True: input & output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        
        
    def create_emb_layer(self,weights_matrix, non_trainable=False):
        # num_embeddings, embedding_dim = #weights_matrix.size()
        emb_layer = torch.absnn.Embedding(self.voab_sz, self.text_emb_sz) #(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer
    
    def load_GLOVE(self,glove_path):
        vectors_path  =f'{glove_path}/6B.50.dat'
        words_path    =f'{glove_path}/6B.50_words.pkl'
        word2idx_path =f'{glove_path}/6B.50_idx.pkl'

        if  not os.path.exists(vectors_path) or \
            not os.path.exists(words_path) or  os.path.exists(word2idx_path): 
                self.proccess_GLOVE(glove_path)
        
        vectors  = bcolz.open(f'{glove_path}/6B.50.dat')[:]
        words    = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
        glove = {w: vectors[word2idx[w]] for w in words}
        return glove

    def proccess_GLOVE(self,glove_path):
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

        with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
    
        vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

    def getTarget_Vocab(self,dataset_name):
        

        Target_dictionary=set()
        if  dataset_name=='mscoco': 
            captions_json_path='./xaigan/src/evaluation/captions_val2014.json'
            with open(captions_json_path) as f:
                captions = json.load(f)
                for caption in captions['annotations']:  
                    Target_dictionary.update(caption['caption'].strip().split())
                    
        elif dataset_name=='flowers-102': 
            captions_json_path='./xaigan/src/evaluation/flowers102_captions.json'
            with open(captions_json_path) as f:
                captions_array = json.load(f).values()
                for Captions in captions_array: 
                    for caption in Captions:
                        Target_dictionary.update(caption.strip().split())               
        return list(Target_dictionary)

    def create_emb_layer_for_Target_Vocab(self,dataset_name,glove_path):

        glove = self.load_GLOVE(glove_path)
        target_vocab = self.getTarget_Vocab(dataset_name) # returns array of words in our dictionary
        self.voab_sz  = len(target_vocab) #number of words
        weights_matrix = np.zeros((self.voab_sz, self.text_emb_sz))
        self.words_found_counter = 0
        self.words_not_found_counter = 0 
        self.words_not_found_list=[]

        for i, word in enumerate(target_vocab):
            try: 
                weights_matrix[i] = glove[word]
                self.words_found += 1
            except KeyError:
                self.words_found_counter+=1
                self.words_not_found_list.append(word)
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.text_emb_sz, ))
        
        return self.create_emb_layer(weights_matrix, non_trainable=False)
    
    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)
    
    def init_hidden(self, batch_size):
        return torch.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
