# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Word to Pix cross-attention via transformer decoder.
Implement on top of DETR.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .visionEnc import build_visionEnc, _get_clones, _get_activation_fn
from .lanEnc import build_lanEnc
from .position_encoding import mypos_sine, mypos_sine1d

class W2P(nn.Module):
    """ Word to Pix cross-attention via transformer decoder """
    def __init__(self, backbone, visionEnc, lanEnc, 
                num_classes, num_attr, num_queries, vocab_size, 
                nhead=8, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1, activation="relu",
                use_pretrained_lfeat=True,learn_query_pos_embedding=True):

        super().__init__()
        self.backbone = backbone
        self.visionEnc = visionEnc
        self.lanEnc = lanEnc
        # bert_dim = 256
        hidden_dim = visionEnc.d_model
        # bert_dim = 768
        bert_dim = lanEnc.bert_dim

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.num_queries = num_queries
        self.use_pretrained_lfeat = use_pretrained_lfeat
        self.learn_query_pos_embedding = learn_query_pos_embedding

        #attribute multi-class loss
        self.attr_mclass_embed = nn.Linear(hidden_dim, num_attr)
        self.sigmoid = nn.Sigmoid()

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self._reset_parameters()


    def forward(self, samples: NestedTensor,input_word_ids, input_word_masks, sent_str: Optional[str] = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - input_word_ids: batched language features of shape [batch_size x sent_len x feat_dim]
               - input_word_masks: mask padded tokens for batch training of shape [batch_size x num_queries]
               - sent_str: single free text query string for inference 

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "outputs_mclass_prob": the multi-class classification score for all attributes.
                                Shape= [batch_size x num_queries x num_attrs]               
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        vis_feat = self.visionEnc(src=self.input_proj(src), mask=mask, pos_embed=pos[-1])

        lan_feat, query_pos_embedding = self.lanEnc(input_word_ids, input_word_masks, sent_str)

        hs = self.decoder(lan_feat, vis_feat, 
            tgt_key_padding_mask=None,
            memory_key_padding_mask=mask.flatten(1), 
            pos=None, 
            query_pos=query_pos_embedding).transpose(1, 2)

        #hs size: [6,bs,que_len,hidden_dim_dec(256)]

        outputs_mclass_prob = self.sigmoid(self.attr_mclass_embed(hs))
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        out = {'outputs_mclass_prob':outputs_mclass_prob[-1], 'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            output = self.norm(output)

        return output.unsqueeze(0)



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.d_model=d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class LossCriterion(nn.Module):
    def __init__(self, num_classes, weight_dict, eos_coef, losses):
        """ Loss terms.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.BCELoss = nn.BCELoss()
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef    #0.1
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # first token for prediction <CLS>
        src_logits = outputs['pred_logits'][:,0]
        target_classes = torch.tensor(targets["cls_labels"], dtype=torch.int64, device=src_logits.device)

        loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits, target_classes)[0]
        return losses

    def attr_loss_labels(self, outputs, targets, log=True):
        """multi-class classfication loss (BCE loss)
        """
        assert 'outputs_mclass_prob' in outputs
        src_probs = outputs['outputs_mclass_prob'][:,0]
        target_labels = torch.tensor(targets["attr_labels"], dtype=torch.float, device=src_probs.device)
        wt_pos=target_labels.sum(dim=1)+1e-3
        wt_neg=(1-target_labels).sum(dim=1)

        loss_BCE = -((src_probs.log()*target_labels).sum(dim=1)/wt_pos + \
            ((1-src_probs).log()*(1-target_labels)).sum(dim=1)/wt_neg).mean()
        # loss_BCE = self.BCELoss(src_probs, target_labels)
        if torch.isnan(loss_BCE) or torch.isinf(loss_BCE):
            loss_BCE = torch.zeros_like(loss_BCE)*src_probs.log().sum()
        losses = {'loss_BCE': loss_BCE}
        return losses

    def loss_boxes(self, outputs, targets, num_boxes=1):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'][:,0]
        target_boxes = torch.tensor(targets["gt_boxes"], dtype=src_boxes.dtype, device=src_boxes.device)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'attr_labels': self.attr_loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def buildW2P(args):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    #COCO no. classes
    num_classes = 91
    device = torch.device(args.device)
    backbone = build_backbone(args)
    visionEnc = build_visionEnc(args)
    lanEnc = build_lanEnc(args)
    model = W2P(
        backbone,
        visionEnc,
        lanEnc,
        num_classes=num_classes,
        num_attr=50,
        num_queries=args.num_queries,
        vocab_size=args.vocab_size,
        num_decoder_layers=args.dec_layers,
        use_pretrained_lfeat = args.use_pretrained_lfeat,
        learn_query_pos_embedding=args.learn_query_pos_embedding,
    )
    weight_dict = {'loss_ce': 1, 
            'loss_BCE': args.attr_loss_coef,
            'loss_bbox': args.bbox_loss_coef,
            'loss_giou': args.giou_loss_coef
            }

    losses = ['labels', 'boxes','attr_labels']

    criterion = LossCriterion(num_classes, 
        weight_dict=weight_dict, 
        eos_coef=args.eos_coef, 
        losses=losses)
    criterion.to(device)

    return model, criterion
