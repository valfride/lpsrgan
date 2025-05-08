from datasets import register
from torch.utils.data import Dataset
from pathlib import Path

@register('multi_image')
class multi_image(Dataset):
    def __init__(self, path_split, phase='training'):

        self.split_file = path_split
        self.phase =  phase
        self.dataset = []
        
        with open(self.split_file, 'r') as f:
            data = f.readlines()
        
        for path in data:
            gt, path_imgs, split = path.split(';')
            path_imgs = path_imgs.strip()
            
            sample = {'gt': gt,
                      'imgs': Path(path_imgs)   
                      }    
            
            if self.phase in split:
                self.dataset.append(sample)
                    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return self.dataset[index]
        
            
            
