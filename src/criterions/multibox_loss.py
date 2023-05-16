import math
import mlconfig

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

@mlconfig.register
class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes=21):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes

    def hard_negative_mining(self, conf_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.
        Args:
          conf_loss: (tensor) cross entroy loss between conf_preds and conf_targets, sized [N*8732,].
          pos: (tensor) positive(matched) box indices, sized [N,8732].
        Return:
          (tensor) negative indices, sized [N,8732].
        '''
        batch_size, num_boxes = pos.size()
        conf_loss[pos.view(-1)] = 0  # set pos boxes = 0, the rest are neg conf_loss
        conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]

        _,idx = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _,rank = idx.sort(1)  # [N,8732]

        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3*num_pos, max=num_boxes-1)  # [N,1]

        neg = rank < num_neg.unsqueeze(1).expand_as(rank) # （N,num_anc_8732)

        return neg

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        '''Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, 8732, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, 8732, 4].
          conf_preds: (tensor) predicted class confidences, sized [batch_size, 8732, num_classes].
          conf_targets: (tensor) encoded target classes, sized [batch_size, 8732].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(conf_preds, conf_targets).
        '''
        batch_size, num_boxes, _ = loc_preds.size()
        pos = conf_targets > 0  # [N,8732], pos means the box matched.
        num_matched_boxes = pos.data.float().sum()
        if num_matched_boxes == 0:
            return torch.tensor([0], requires_grad=True, dtype=torch.float32)
        
        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)    # [N,8732,4]
        pos_loc_preds = loc_preds[pos_mask].view(-1,4)      # [#pos,4]
        pos_loc_targets = loc_targets[pos_mask].view(-1,4)  # [#pos,4]
        
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)

        ################################################################
        # conf_loss = CrossEntropyLoss(pos_conf_preds, pos_conf_targets)
        #           + CrossEntropyLoss(neg_conf_preds, neg_conf_targets)
        ################################################################
        conf_loss = F.cross_entropy(conf_preds.view(-1,self.num_classes), \
                                            conf_targets.view(-1), reduce=False)  # [N*8732,]
        neg = self.hard_negative_mining(conf_loss, pos)    # [N,8732]

        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        mask = (pos_mask+neg_mask).gt(0)

        pos_and_neg = (pos+neg).gt(0)
        preds = conf_preds[mask].view(-1,self.num_classes)  # [#pos+#neg,21]
        targets = conf_targets[pos_and_neg]                 # [#pos+#neg,]
        conf_loss = F.cross_entropy(preds, targets, size_average=False)

        loc_loss /= num_matched_boxes
        conf_loss /= num_matched_boxes

        return loc_loss + conf_loss