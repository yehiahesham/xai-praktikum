import torch
import os
import random
import logging
import numpy as np
from torch import cuda
from transformers import RobertaModel, RobertaTokenizer


class RobertaClass(torch.nn.Module):
    def __init__(self, max_len=350, use_CLS_emb=True,use_one_caption=True ):
        super().__init__()
        #super(RobertaClass, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        self.use_CLS_emb=use_CLS_emb # 2 Options to get embeddings : CLS Token or Pool RoBERTa Output
        self.use_one_caption=use_one_caption #Bool to either use all image's captions or just one (the 1st).
        self.max_len = max_len
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.__set_seed__(seed = 318)
        logging.basicConfig(level=logging.ERROR)

    def __set_seed__(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True

    def __get_ImageCaption_Tokens__(self,image_captions,):
        if self.use_one_caption :
            texts = image_captions[0] #pick 1st caption image
        else :
            texts = image_captions # Use all caption/image

        tokenized = self.tokenizer.encode_plus(
            texts,
            truncation=True,         #hyperparameter
            add_special_tokens=True, #hyperparameter
            max_length=self.max_len, #hyperparameter
            padding="max_length"
        )
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]
        #token_type_ids = tokenized["token_type_ids"] #we could also use that. for simplicity commented.
        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            "token_type_ids": None, #torch.tensor(token_type_ids, dtype=torch.long),
        }
        
    def __get_Roberta_output__(self,tokenizer_output):
        '''
        Desc:
            Function Tokenizes the caption string(s) and pass it to roberta model and return its outputs. 

        Inputs:
        -tokenizer_output: dict. of the tokenizer output {ids,mask,token_type_ids}
        
        -use_one_caption: bool
            flag to consider either all image's captions or just the 1st

        Outputs:
        -last_hidden_state : []
            1st Roberta output which shape: (#strings, max_len, embedding_size)

        -pooler_output: []
            2nd  Roberta output which shape (#strings, max_len, embedding_size)
        '''

        #A) get ids,mask,token_type_ids from the tokenization
        ids =tokenizer_output["ids"].to(self.device).unsqueeze(0)
        mask=tokenizer_output["mask"].to(self.device).unsqueeze(0)
        #token_type_ids = tokenized["token_type_ids"].unsqueeze(0) #Not needed.

        #B) Run robera model with the tokens's ids and mask
        text_model_output = self.roberta(ids, mask,token_type_ids=None) # get embddeding
        #C) get robeta output, ie last_hidden_state & pooler output
        last_hidden_state = text_model_output [0] # or text_model_output["last_hidden_state"]
        pooler_output     = text_model_output [1] # or text_model_output["pooler_output"]

        #D) return Roberta Output
        return last_hidden_state ,pooler_output

    def __get_Robertra_cls_embeddings__(self,last_hidden_state ,pooler_output):
        cls_embeddings = last_hidden_state[:, 0, :].detach()
        return cls_embeddings

    def __get_Robertra_pool_embeddings__(self,last_hidden_state ,pooler_output): 
        '''
        note!: pooler output "not" equal pooled_embeddings we calculated  
        What is pooler output ?  
        -> It takes the representation from the [CLS] token from top layer of RoBERTa encoder, and feed that through another dense layer.  
        reference: https://github.com/google-research/bert/blob/cc7051dc592802f501e8a6f71f8fb3cf9de95dc9/modeling.py#L224-L232  
        '''
        pooled_embeddings = last_hidden_state.detach().mean(dim=1)
        return pooled_embeddings

    def get_ImageCaptions_embedding(self,last_hidden_state ,pooler_output,image_captions=None):
        
        #A) get Roberta Output if needed
        if last_hidden_state==None and pooler_output==None:    
            last_hidden_state ,pooler_output = self.__get_Roberta_output__(self,image_captions)

        #B) return desired embedding 
        if self.use_CLS_emb :
            return self.__get_Robertra_cls_embeddings__(last_hidden_state ,pooler_output)
        else:
            return self.__get_Robertra_pool_embeddings__(last_hidden_state,pooler_output)
        
    def forward(self,image_captions):
        '''
        Desc: 
            Returns text(s) embeddings
        
        Inputs:
            -image_captions: [str], array of text captions desc. the image 

        Outputs:
            based on the embedding choice [cls or pool], we return the text(s) embedding(s)
        '''
        tokenizer_output = self.__get_ImageCaption_Tokens__(image_captions)
        last_hidden_state ,pooler_output = self.__get_Roberta_output__(tokenizer_output)

        return self.get_ImageCaptions_embedding(last_hidden_state ,pooler_output)

    
if __name__=='__main__':
    text_emb_model = RobertaClass(max_len=350, use_CLS_emb=True,use_one_caption=False)
    texts = [
        "Saw met applauded favourite deficient engrossed concealed and her",
        "Egyptian man fights aliens"
         ]
    # texts = ["Saw met applauded favourite deficient engrossed concealed and her"]
    texts_emb = text_emb_model.forward(texts)
    print(texts_emb.shape)

