# label form: 'XXX.jpg object_sum x1 y1 x2 y2 class\n'
import mlconfig
import mlflow
import numpy as np
import torch
from src.datasets.encoder import DataEncoder

import src


config = mlconfig.load('configs/config.yaml')
mlflow.log_artifact('configs/config.yaml')
mlflow.log_params(config.flat())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = config.model().to(device)
model = torch.load('./model/last.pt')
test_loader = config.dataset(root='./data/test', list_file='./data/test.txt', train=True)

if __name__ == "__main__":
  scale = 512
  aspect_ratios = [[2,], [2,3], [2,3], [2,3], [2,], [2,]]
  feature_map = [64, 32, 16, 8, 4, 2]
  sizes = [25, 50, 75, 100, 125, 150, 200]
  data_decoder = DataEncoder(scale, aspect_ratios, feature_map, sizes)
  with torch.no_grad():
      for x, loc_targets, conf_targets in self.test_loader:
          x = x.to(self.device)
          loc_targets = loc_targets.to(self.device)
          conf_targets = conf_targets.to(self.device)
  
          loc_preds, conf_preds = self.model(x)
          
          data_decoder.decode(loc_preds, conf_preds)
