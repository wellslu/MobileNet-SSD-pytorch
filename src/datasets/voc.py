import mlconfig
import torchvision.transforms as transforms
from torch.utils import data
from .datagen import ListDataset


@mlconfig.register
class VOCDataLoader(data.DataLoader):

    def __init__(self, root: str, list_file: str, train: bool, batch_size: int, scale: int, aspect_ratios: list, feature_map: list, sizes: list, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((scale,scale)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        dataset = ListDataset(root, list_file, transform, scale, aspect_ratios, feature_map, sizes)

        super(VOCDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)
