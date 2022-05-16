import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import math
import sys
from pprint import pprint
from collections import OrderedDict
import numpy as np
import torch
import util.misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from torch.autograd import Variable
import util.misc as utils
import datasets
from datasets.ref_loader import RefDataset
from datasets import build_dataset, get_coco_api_from_dataset
from models.w2p import buildW2P
from typing import Iterable
from util.misc import NestedTensor, nested_tensor_from_tensor_list


def main():
    parser = argparse.ArgumentParser(description='Word2Pix VG Training Script')

    # * Training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=350, type=int)
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--save_every', default=10, type=int)    
    parser.add_argument('--pretrained', default=True, type=bool,help="use pretrained backbone")
    parser.add_argument('--finetune', default=None , help='finetune from checkpoint')
    parser.add_argument('--resume', default=None , help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # * Dataset parameters
    parser.add_argument('--dataset_split', default='refcoco_unc',choices=('refcoco_unc','refcoco+_unc','refcocog_umd'))
    parser.add_argument('--bert_feat_rt', default='./prepro/', help="pretrained bert feature for ref sentences")
    parser.add_argument('--seq_per_ref', default=3, type=int, help="Number of sentences per refered object for training")
    parser.add_argument('--coco_path', default=r'./cocopth', type=str)

    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    parser.add_argument('--num_queries', default=10, type=int, help="Number of text query tokens")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--vocab_size', default=2000, type=int, help="vocab_size, embedding LUT size")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--use_pretrained_lfeat', default=True, type=bool,help="use cached pre-extracted language feature") 

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
    parser.add_argument('--num_workers', default=12, type=int)

    args = parser.parse_args()
    utils.init_distributed_mode(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # dataset_rt='../MAttNet/cache/prepro/'
    #'refcocog_umd','refcoco_unc','refcoco+_unc'

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #DATASET
    dataset_train = RefDataset(args, seq_per_ref=args.seq_per_ref,train_val='train')
    dataset_val = RefDataset(args, seq_per_ref=1, train_val='val')
    
    args.num_queries = 15 if args.dataset_split == 'refcocog_umd' else 10
    args.vocab_size = int(dataset_train.vocab_size)

    args.finetune = r'../chkp/refcoco/checkpoint0084.pth'

    print(args)
    device = torch.device(args.device)
    model, criterion = buildW2P(args)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    elif args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'],strict=True) 
    else:
        if args.pretrained:
            # use pretrained CNN+encoder weights excluding refcoco/+/g val/test
            checkpoint = torch.load('./chkp/detr_r101_250ep_refcoco_exc.pth', map_location='cpu')
            # use pretrained CNN+encoder weights excluding non-refcoco/+/g imgs
            checkpoint = torch.load('./chkp/detr_r101_150ep_refcoco_only.pth', map_location='cpu')
            curr_p=[n for n, _ in model_without_ddp.named_parameters()]
            ckpt_p=[n for n in checkpoint['model'].keys()]
            for n in ckpt_p:
                if 'query_embed' in n or 'decoder' in n: 
                    del checkpoint['model'][n]
        else:
            # use pretrained CNN backbone only
            pretrained_backbone_coco='./chkp/bkb_coco.pth'
            checkpoint = torch.load(pretrained_backbone_coco, map_location='cpu')
            exc_list=['fc','rpn','bbox','cls','mask']
            md=OrderedDict()
            for n,p in checkpoint.items():
                n1=n.replace('resnet','backbone.0.body')
                flg=False
                for i in exc_list:
                    if i in n1:
                        flg=True
                        break
                if flg:
                    continue
                else:
            #         print(n1)
                    md[n1]=p
            checkpoint = md   
        model_without_ddp.backbone.load_state_dict(checkpoint,strict=False)


    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.w2p_coll, num_workers=args.num_workers, pin_memory=True)

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, max_norm=0.1)

        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or epoch % save_every == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for img, sent, sent_mask, cls_lb, boxes, attr_lb in metric_logger.log_every(data_loader, print_freq, header):
            
        boxes=torch.from_numpy(boxes)
        input_word_ids = torch.from_numpy(sent).to(device)
        input_word_masks = torch.from_numpy(sent_mask).long().to(device)

        outputs = model(img.to(device),input_word_ids, input_word_masks)
        loss_dict = criterion(outputs, {'cls_labels':cls_lb,'gt_boxes':boxes,'attr_labels':attr_lb})
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            continue

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



if __name__ == '__main__':
    main()
