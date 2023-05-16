import os
from PIL import Image
import numpy as np
import cv2
import torch
import torch.utils.data as data

from .encoder import DataEncoder



class ListDataset(data.Dataset):

    def __init__(self, root, list_file, transform, scale, aspect_ratios, feature_map, sizes):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder(scale, aspect_ratios, feature_map, sizes)

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])

            box = []
            label = []
            if len(splited) > 1:
                num_objs = int(splited[1])
                for i in range(num_objs):
                    xmin = splited[2+5*i]
                    ymin = splited[3+5*i]
                    xmax = splited[4+5*i]
                    ymax = splited[5+5*i]
                    c = splited[6+5*i]
                    box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                    label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load a image, and encode its bbox locations and class labels.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          loc_target: (tensor) location targets, sized [8732,4].
          conf_target: (tensor) label targets, sized [8732,].
        '''
        # Load image and bbox locations.
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        # Scale bbox locaitons to [0,1].
        w,h = img.shape[1], img.shape[0]
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.transform(img)

        # Encode loc & conf targets.        
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target


    def __len__(self):
        return self.num_samples
