from utils import *
import glob
from torchvision import transforms
from tqdm import trange
import random

# Spike Dataset
class SpikeData(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels, stage):
        self.root_dir = root_dir
        self.stage = stage
        self.data_list = os.path.join(root_dir, stage)
        self.data_list = sorted(os.listdir(self.data_list))
        self.labels = labels
        self.length = len(self.data_list)
    
    def __getitem__(self, idx: int):
        data = np.load(os.path.join(self.root_dir,self.stage,self.data_list[idx]))
        
        # Defensive loading: check required keys
        if 'spk' not in data or 'label' not in data:
            raise ValueError(f"Missing keys in {self.data_list[idx]} (found: {list(data.keys())})")
        
        spk = data['spk'].astype(np.float32)
        
        # Dynamic center crop to 224x224 (works for any input size)
        # Robust to variable T (number of frames)
        T, H, W = spk.shape
        crop_h, crop_w = 224, 224
        top = max(0, (H - crop_h) // 2)
        left = max(0, (W - crop_w) // 2)
        spk = spk[:, top:top+crop_h, left:left+crop_w]  # [T, 224, 224]
        
        label_idx = int(data['label'])
        label = self.labels[label_idx]
        return spk,label,label_idx

    def __len__(self):
        return self.length

if __name__ == "__main__":
    labels = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background']
    dataset = SpikeData('data/U-CALTECH',labels,'train')
    spk,label,label_idx = dataset[0]
    print(spk.shape,label,label_idx)

