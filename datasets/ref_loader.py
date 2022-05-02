"""
data_json has 
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import h5py
import json
import pickle
import random
from datasets import build_dataset

import torch

# index: [img_id, ref_id]
class RefDataset(torch.utils.data.Dataset):
    def __init__(self,
        # data_json,
        args,
        # data_h5=None,#tokenized words capped to max len 10
        seq_per_ref=3,
        use_pretrained_lfeat=True,
        train_val='train'):

        data_json = osp.join(args.bert_feat_rt,args.dataset_split,'data.json')
        data_h5 = osp.join(args.bert_feat_rt,args.dataset_split,'data.h5')
        
        print('Loader loading data.json: ', data_json)
        self.info = json.load(open(data_json))
        self.word_to_ix = self.info['word_to_ix']
        #ADD <CLS> token
        self.word_to_ix.update({'<CLS>':max(self.word_to_ix.values())+1})
        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        print('vocab size is ', self.vocab_size)
        self.cat_to_ix = self.info['cat_to_ix']
        self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
        print('object cateogry size is ', len(self.ix_to_cat))
        self.images = self.info['images']
        self.anns = self.info['anns']
        self.refs = self.info['refs']
        self.sentences = self.info['sentences']
        print('we have %s images.' % len(self.images))
        print('we have %s anns.' % len(self.anns))
        print('we have %s refs.' % len(self.refs))
        print('we have %s sentences.' % len(self.sentences))
        print('label_length is ', self.label_length)

        # construct mapping
        self.Refs = {ref['ref_id']: ref for ref in self.refs}
        self.Images = {image['image_id']: image for image in self.images}
        self.Anns = {ann['ann_id']: ann for ann in self.anns}
        self.Sentences = {sent['sent_id']: sent for sent in self.sentences}
        self.annToRef = {ref['ann_id']: ref for ref in self.refs}
        self.sentToRef = {sent_id: ref for ref in self.refs for sent_id in ref['sent_ids']}
        
        # read data_h5 if exists: 
        #data_h5 stores tokenized words of all sents capped at label_length(refg:15,other:10)
        self.data_h5 = None
        if data_h5 is not None:
          #if .h5 file invalid, convert to .pkl
          if data_h5.endswith('.h5'):
            with h5py.File(data_h5, 'r') as f:
              self.data_h5_arr = np.array(f['labels'])
            with open(data_h5.replace('.h5','.pkl'),'wb') as f:
              pickle.dump(self.data_h5_arr,f)
          else:
            print('Loader loading data.pkl: %s', data_h5)
            with open(data_h5, 'rb') as f:
              self.data_h5_arr = pickle.load(f)

        assert self.data_h5_arr.shape[0] == len(self.sentences), 'label.shape[0] not match sentences'
        assert self.data_h5_arr.shape[1] == self.label_length, 'label.shape[1] not match label_length'
        
        self.coco_train, self.coco_test=build_dataset(image_set='train', coco_path=args.coco_path,train_mode=args.pretrained)
        refId_to_imgBoxId_fn = osp.join(args.refId_to_imgBoxId_path, 'refId_to_imgBoxId_%s.npy'%args.dataset_split)
        self.refId_to_imgBoxId = np.load(refId_to_imgBoxId_fn, allow_pickle=True).item()
        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.num_atts = len(self.att_to_ix)
        self.att_to_cnt = self.info['att_to_cnt']

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for image_id, image in self.Images.items():
          # we use its ref's split (there is assumption that each image only has one split)
          split = self.Refs[image['ref_ids'][0]]['split']
          if split not in self.split_ix:
            self.split_ix[split] = []
            self.iterators[split] = 0
          self.split_ix[split] += [image_id]
        for k, v in self.split_ix.items():
          print('assigned %d images to split %s' % (len(v), k))

        self.list_IDpairs=list()
        for image_id in self.split_ix[train_val]:
          for ref_id in self.Images[image_id]['ref_ids']:
            self.list_IDpairs.append([image_id,ref_id])
        print('# of [img,ref] pairs in %s: %d' %(train_val, len(self.list_IDpairs)))

        self.seq_per_ref = seq_per_ref
        #use BERT setting
        self.use_pretrained_lfeat = use_pretrained_lfeat
        self.feat_dim = 768
        #refcocox dataset: [Ntok, 768]: [CLS],tk0,tk1,...
        self.bert_feat_pth = osp.join(args.bert_feat_rt,args.dataset_split,'sentid2bert_feat')
        self.feat_len = self.label_length
        self.train_val = train_val

    @property
    def vocab_size(self):
        return len(self.word_to_ix)

    @property
    def label_length(self):
        return self.info['label_length']    

    def fetch_seq(self, sent_id):
        # return int32 (label_length, )
        sent_h5_id = self.Sentences[sent_id]['h5_id']
        seq = self.data_h5_arr[sent_h5_id, :]
        #prepend <CLS> token
        seq = np.insert(seq[:-1],0,self.word_to_ix['<CLS>'])
        return seq

  # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def resetIterator(self, split):
        self.iterators[split] = 0

    # expand list by seq_per_ref, i.e., [a,b], 3 -> [aaabbb]
    def expand_list(self, L, n):
        out = []
        for l in L:
          out += [l] * n
        return out

    def fetch_sent_ids_by_ref_id(self, ref_id, num_sents, train_val='train'):
      """
      Sample #num_sents sents for each ref_id.
      """
      if train_val=='train':
        sent_ids = list(self.Refs[ref_id]['sent_ids'])
        if len(sent_ids) < num_sents:
          append_sent_ids = [random.choice(sent_ids) for _ in range(num_sents - len(sent_ids))]
          sent_ids += append_sent_ids
        else:
          random.shuffle(sent_ids)
          sent_ids = sent_ids[:num_sents]
        assert len(sent_ids) == num_sents
      else:
        sent_ids = list(self.Refs[ref_id]['sent_ids'])
      return sent_ids

    def __len__(self):
        return len(self.list_IDpairs)


    def __getitem__(self,index):
        # return 3 sentences pertaining to one [img, refid] pair
        # each time img is transformed differently
        image_id = self.list_IDpairs[index][0]
        ref_id = self.list_IDpairs[index][1]
        #attr labels targets
        attr_labels = np.zeros(self.num_atts)
        wds = self.Refs[ref_id]['att_wds']
        if len(wds) > 0:
          for wd in wds:
            attr_labels[self.att_to_ix[wd]] = 1
                
        pos_sent_ids = self.fetch_sent_ids_by_ref_id(ref_id, self.seq_per_ref, train_val=self.train_val)

        ref_ann_id = self.Refs[ref_id]['ann_id']
        idx = self.refId_to_imgBoxId[ref_id]
        if self.train_val == 'train':
          img_from_coco = self.coco_train.__getitem__(image_id)
        else:
          #same set, data aug transform is different
          img_from_coco = self.coco_test.__getitem__(image_id)

        sample = list()
        # bert feats
        if self.use_pretrained_lfeat:
          sents = []
          sents_mask = []
          for i, sent_id in enumerate(pos_sent_ids):
            sent = torch.zeros(1,self.feat_len,self.feat_dim)
            sent_mask = [True]*self.feat_len
            with open(osp.join(self.bert_feat_pth,'%d.pkl'%sent_id),'rb') as f:
              #cap at 10 tokens. note bert tokens are longer than ntokens
              feat = pickle.load(f)[:self.feat_len,:]
              sent[0,:len(feat),:] = feat
              sents.append(sent)
              sent_mask[:len(feat)] = [False]*len(feat)
              sents_mask.append(sent_mask)
        # word tokens
        #TODO?:add inference pipeline from BERT
        else:
          sents = [self.fetch_seq(sent_id) for sent_id in pos_sent_ids]
          sents_mask = [sent==0 for sent in sents]

        for sent, sent_mask in zip(sents, sents_mask):  
          sample.append(np.array([img_from_coco[0],\
            sent,\
            sent_mask,\
            img_from_coco[1]['labels'][idx],\
            img_from_coco[1]['boxes'][idx],\
            attr_labels],dtype=object))

        sample = np.vstack(sample)
        return sample



