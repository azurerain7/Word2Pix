#!/usr/bin/env python
from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import argparse
import sys
import numpy as np
import json
import h5py
import time
import random
from pprint import pprint
from PIL import Image

from datasets import build_dataset
import datasets
from models.w2p import buildW2P
from datasets.ref_loader import RefDataset
import util.misc as utils
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.misc import iou
from util import box_ops
from util.plot_utils import show_res

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Cache preprocessed dataset text features.')
    
    # * Dataset parameters
    parser.add_argument('--dataset_split', default='refcoco_unc',choices=('refcoco_unc','refcoco+_unc','refcocog_umd'))
    parser.add_argument('--coco_path', default=r'./cocopth', type=str)
    parser.add_argument('--bert_feat_rt', default='./prepro/', help="pretrained bert feature for ref sentences")
    parser.add_argument('--pretrained', default=True, type=bool,help="use pretrained backbone")
    parser.add_argument('--refId_to_imgBoxId_path', default='./datasets')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)


    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device=args.device
    
    print('Caching text feature for %s:'%args.dataset_split)

    dataset_val = RefDataset(args, train_val='val')
    
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,)
    text_model.to(device)
    text_model.eval()
    
    data_loader = dataset_val
    #length of sents are decided by preprocessing in MAttnet: refcoco/+: 10, refcocog: 20
    sent2txt_bert={}
    for ref in data_loader.Refs.values():
        for sent_id in ref['sent_ids']:
            seq = data_loader.data_h5_arr[data_loader.Sentences[sent_id]['h5_id'],:]
            sent2txt_bert[sent_id] = '[CLS] '+' '.join([data_loader.ix_to_word[wd] for wd in np.delete(seq,seq==0)])+' [SEP]'
    
    
    import pickle
    rt = os.path.join('./prepro',args.dataset_split,'sentid2bert_feat')
    N=len(sent2txt_bert)
    n=0
    # sent2feat_bert={}
    with torch.no_grad():
        for sent_id, tt in sent2txt_bert.items():
            tokenized_text = tokenizer.tokenize(tt)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1]*len(tokenized_text)
            outputs = text_model(torch.tensor([indexed_tokens]).to(device),
                                 torch.tensor([segments_ids]).to(device))
            hidden_states = outputs[2]
    #         sent2feat_bert[sent_id] = hidden_states[-1][0][:-1]
            v = hidden_states[-1][0][:-1].cpu()
            #saving to file
            with open(os.path.join(rt,'%d.pkl'%sent_id),'wb') as f:
                pickle.dump(v,f)

            pbar_str='\r progress: %d/%d\t'%(n,N)
            sys.stdout.write(pbar_str)
            sys.stdout.flush()
            n+=1

if __name__ == '__main__':
    main()