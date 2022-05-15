"""
LanEncdoer:
Use pretrained BERT as language encoder.
For batched training/testing:
    language features cached to files and loaded by dataset.
For inference with single free text query:
    BERT model is loaded&init from huggingface: transformers lib. 
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .position_encoding import mypos_sine, mypos_sine1d
from transformers import BertTokenizer, BertModel


class LanEncoder(nn.Module):

    def __init__(self, num_queries, vocab_size, hidden_dim=256, bert_dim=768,
        use_pretrained_lfeat=True,learn_query_pos_embedding=True):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.bert_dim = bert_dim
        self.use_pretrained_lfeat = use_pretrained_lfeat
        self.learn_query_pos_embedding = learn_query_pos_embedding

        if not self.use_pretrained_lfeat:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_model = BertModel.from_pretrained('bert-base-uncased', 
                output_hidden_states=True)
            self.text_model.eval()

        self.query_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_word_proj = nn.Linear(self.bert_dim, hidden_dim)

    def forward(self, sent, sent_mask, sent_str: Optional[str] = None):
        #TRAINING/TESTING: from dataset: input: sent, sent_mask
        #FREE INFERENCE: with input query text string
        
        if self.use_pretrained_lfeat:
            # FOR BATCHED TRAINING AND TESTING
            input_word_ids = sent
            input_word_masks = sent_mask
            m = (input_word_masks.sum(dim=0)!=input_word_masks.size(0))
            bsz = input_word_masks.size(0)
            dim = input_word_ids.size(2)
            msk = m.repeat(input_word_masks.size(0),1)
            
            input_word_ids = input_word_ids[msk].view(bsz,-1,dim)[:,:self.num_queries,:]
            input_word_masks = input_word_masks[msk].view(bsz,-1).bool()[:,:self.num_queries]
                
            query_inp = self.input_word_proj(input_word_ids)
            query_mask = input_word_masks
            '''
            if not self.use_pretrained_lfeat:
                #legacy: learn text representation from scratch
                query_inp = self.query_embed(input_word_ids)
                query_mask = None 
                pos_ids=torch.arange(self.num_queries,dtype=torch.long,device=input_word_ids.device)
                pos_ids = pos_ids.unsqueeze(0).expand_as(input_word_ids)
            '''
        else:
            #FOR INFERENCE WITH FREE TEXT QUERY
            assert sent_str is not None
            tt = '[CLS] '+sent_str+ ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(tt)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1]*len(tokenized_text)
            with torch.no_grad():
                outputs = self.text_model(torch.tensor([indexed_tokens]), torch.tensor([segments_ids]))
                hidden_states = outputs[2]
            # print(tt)
            # print(tokenized_text)
            # LAST layer of BERT representation
            input_word_ids=hidden_states[-1][0][:-1].unsqueeze(0)
            # print(input_word_ids.shape)
        
            query_inp = self.input_word_proj(input_word_ids)
            #query_mask only used for posEmb
            if len(indexed_tokens) < self.num_queries:
                query_mask = torch.tensor([[True]*len(self.num_queries)])
                query_mask[0][:len(indexed_tokens)] = [False]*len(indexed_tokens)
            else:
                query_mask = torch.tensor([[False]*len(indexed_tokens)])

        pos_ids = torch.arange(input_word_ids.size(1),dtype=torch.long,device=input_word_ids.device)
        pos_ids = pos_ids.repeat(input_word_ids.size(0),1)

        #False for scratch 
        #learn_query_pos_embedding=False #for rebuttal exp
        #learn_query_pos_embedding=True #for previous exp
        if self.learn_query_pos_embedding:
            query_pos_embedding = self.query_pos_embed(pos_ids)
        else:
            query_pos_embedding = mypos_sine1d(query_mask, num_pos_feats=self.hidden_dim)

        lan_feat = query_inp.flatten(2).permute(1, 0, 2)
        if query_pos_embedding is not None:
            query_pos_embedding = query_pos_embedding.flatten(2).permute(1, 0, 2)

        return lan_feat, query_pos_embedding


def build_lanEnc(args):
    return LanEncoder(
        num_queries=args.num_queries,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        use_pretrained_lfeat = args.use_pretrained_lfeat,
        learn_query_pos_embedding=args.learn_query_pos_embedding
    )


