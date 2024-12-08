from torch.utils.data import Dataset
import numpy as np
import json
from bisect import bisect_right

class MahjongGBDataset(Dataset):
    
    def __init__(self, begin=0, end=1):
        with open('data/count.json') as f:
            self.match_samples = json.load(f)
        
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        match_data = self._load_match_data(match_id)
        
        obs = match_data['obs'][sample_id]
        mask = match_data['mask'][sample_id]
        act = match_data['act'][sample_id]
        
        return obs, mask, act
    
    def _load_match_data(self, match_id):
        # Load the specific match data from disk
        data = np.load(f'data/{match_id + self.begin}.npz')
        return data
