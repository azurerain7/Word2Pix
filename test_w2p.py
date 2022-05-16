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
    parser = argparse.ArgumentParser(description='Word2Pix VG Testing Script', add_help=False)

    # * Testing
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--checkpoint', 
    	default=r'/raid/zh/RESEARCH/detr_zh/scra_trainall_704_sq3bt16/checkpoint0084.pth',type=str)
    parser.add_argument('--pretrained', default=True, type=bool,help="use pretrained backbone")

    # * Dataset parameters
    parser.add_argument('--dataset_split', default='refcoco_unc',choices=('refcoco_unc','refcoco+_unc','refcocog_umd'))
    parser.add_argument('--bert_feat_rt', default='../MAttNet/cache/prepro/', help="pretrained bert feature for ref sentences")
    parser.add_argument('--coco_path', default=r'./cocopth', type=str)

    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,help="Name of the convolutional backbone to use")
    parser.add_argument('--learn_query_pos_embedding', default=True, type=bool,help="learn PosEmb instead of fixed sine1d emb for query") 
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_queries', default=10, type=int, help="Number of text query tokens")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--use_pretrained_lfeat', default=True, type=bool,help="use cached pre-extracted language feature") 

    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--vocab_size', default=2000, type=int, help="vocab_size, embedding LUT size")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    # default on detection is 1,5,2 for cls, bbox, giou, mask~1
    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--attr_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,help="Relative classification weight of the no-object class")
    
    #VG
    parser.add_argument('--refId_to_imgBoxId_path', default='./datasets')
    parser.add_argument('--output_dir', default='./outp', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None , help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=12, type=int)

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #REFCOCO
    args.dataset_split = 'refcoco_unc' 
    #PARTIAL INIT: encoder scratch
    #args.pretrained=False
    #args.checkpoint = r'../chpt/refcoco_enc_scratch.pth'#8144/7256/v7738 
    #ENCODER transfer from refcoco det
    #args.checkpoint = r'../chpt/refcoco_enc_trans.pth' #8213/7649/v7898
    #ENCODER INIT: detr encoder
    args.checkpoint = r'../chpt/refcoco_enc_detr.pth' #8439/7812/v8112 
    
    #REFCOCO+
    # args.dataset_split = 'refcoco+_unc' 
    # args.checkpoint = r'../ckpt/refcocop_sota.pth' #7611/6124/v6974

    #REFCOCOg
    # args.dataset_split = 'refcocog_umd' 
    # args.checkpoint = r'../ckpt/refcocogumd_sota.pth'  #7081/7134)

    dataset_test_list = []
    dataset_val = RefDataset(args, train_val='val')

    if args.dataset_split == 'refcocog_umd':
        dataset_test = RefDataset(args, train_val='test')
        dataset_test_list =[dataset_val, dataset_test]
    else:
        dataset_testA = RefDataset(args, train_val='testA')
        dataset_testB = RefDataset(args, train_val='testB')
        dataset_test_list =[dataset_val, dataset_testA, dataset_testB]

    #'refcoco_unc','refcoco+_unc','refcocog_umd'
    if args.dataset_split == 'refcoco_unc':
        args.vocab_size = 2000
        args.num_queries = 10
    elif args.dataset_split == 'refcoco+_unc':
        args.vocab_size = 2633
        args.num_queries = 10
    elif args.dataset_split == 'refcocog_umd':
        args.vocab_size = 3350
        args.num_queries = 15
    else:
        pass
    if args.pretrained:   
        args.batch_size=6
        args.learn_query_pos_embedding=True
    else:
        args.batch_size=1
        args.learn_query_pos_embedding=False

    device = args.device
    model, _ = buildW2P(args)
    model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    for iii, dataset_test in enumerate(dataset_test_list):
        sampler = torch.utils.data.SequentialSampler(dataset_test)
        batch_sampler = torch.utils.data.BatchSampler(sampler,args.batch_size,drop_last=False)
        data_loader = DataLoader(dataset_test, 
                            batch_sampler=batch_sampler,
                            collate_fn=utils.w2p_coll, 
                            num_workers=args.num_workers, 
                            pin_memory=True)

        seq_idx=0
        thd=0.5
        tcnt=0
        pcnt=0
        n=0
        N=len(data_loader)
        for img, sent, sent_mask, cls_lb, boxes, attr_lb in data_loader:
            
            boxes=torch.from_numpy(boxes)
            input_word_ids = torch.from_numpy(sent).to(device)
            input_word_masks = torch.from_numpy(sent_mask).long()

            with torch.no_grad():
                outputs = model(img.to(device),input_word_ids, input_word_masks)

            #check all sentences for one referred object
            for bsi in range(len(outputs['pred_boxes'])):
                pbbn=np.array(box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][bsi][seq_idx].detach()).cpu())
                gbbn=np.array(box_ops.box_cxcywh_to_xyxy(boxes[bsi]))
                iou_score = iou(pbbn,gbbn)
                if iou_score>=thd:
                    pcnt+=1
                tcnt+=1
            
            n+=1
            pbar_str='\r progress: %d/%d\t'%(n,N)
            sys.stdout.write(pbar_str)
            sys.stdout.flush()

        pprint(args.dataset_split)
        pprint(iii)
        pprint(args.checkpoint)
        pprint('accuracy@thd=%.2f: %.4f'%(thd,pcnt/tcnt*1.))


if __name__ == '__main__':
    main()


