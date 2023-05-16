import numpy as np
import torch

class MIOU(object):

    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def get_iou(self, output, target):
        _, pred = torch.max(output, 1)

        if pred.device == torch.device('cuda'):
            pred = pred.cpu()
        if target.device == torch.device('cuda'):
            target = target.cpu()

        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        pred += 1
        target += 1

        pred = pred * (target > 0)
        inter = pred * (pred == target)
        area_inter = torch.histc(inter.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_pred = torch.histc(pred.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_mask = torch.histc(target.float(), bins=self.num_classes, min=1, max=self.num_classes)
        area_union = area_pred + area_mask - area_inter + self.epsilon

        return area_inter.numpy(), area_union.numpy()
