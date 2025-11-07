import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import clip
import numpy as np
import sys
sys.path.append("..")
import argparse
from datetime import datetime
from torch.optim import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils import * 
from dataset import *
from models import *
from torchvision.transforms import Normalize
from metrics import *

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

if __name__ == '__main__':
    # parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default='test')
    parser.add_argument('--data_type', type=str,default='CALTECH') # CALTECH
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default='models/LRN_CALTECH.pth', help='Path to model checkpoint')
    
    opt = parser.parse_args()
    # labels
    if opt.data_type == 'CALTECH':
        labels = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background']
        opt.base_folder = 'data/U-CALTECH'
        opt.save_folder = 'exp/CALTECH_eval'
    elif opt.data_type == 'CIFAR':
        labels =  ["frog","horse","dog","truck","airplane","automobile","bird","ship","cat","deer"]
        opt.base_folder = 'data/U-CIFAR'
        opt.save_folder = 'exp/CIFAR_eval'
        
    # prepare
    ckpt_folder = f"{opt.save_folder}/{opt.exp_name}/ckpts"
    img_folder = f"{opt.save_folder}/{opt.exp_name}/imgs"
    os.makedirs(ckpt_folder,exist_ok= True)
    os.makedirs(img_folder,exist_ok= True)
    set_random_seed(opt.seed)
    save_opt(opt,f"{opt.save_folder}/{opt.exp_name}/opt.txt")
    log_file = f"{opt.save_folder}/{opt.exp_name}/results.txt"
    logger = setup_logging(log_file)
    if os.path.exists(f'{opt.save_folder}/{opt.exp_name}/tensorboard'):
        shutil.rmtree(f'{opt.save_folder}/{opt.exp_name}/tensorboard')
    writer = SummaryWriter(f'{opt.save_folder}/{opt.exp_name}/tensorboard')
    logger.info(opt)

    # train and test data splitting
    train_dataset = SpikeData(opt.base_folder,labels,stage = 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=4,pin_memory=True)
    test_dataset = SpikeData(opt.base_folder,labels,stage = 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1,pin_memory=True)
    
    # network 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(device)
    clip_model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
    clip_model = clip_model.to(device)
    for name, param in clip_model.named_parameters():
        param.requires_grad_(False)
    from models import SNN_LRN_Wrapper
    recon_net = SNN_LRN_Wrapper(inDim=50, outDim=1, num_steps=50).to(device)
    
    # Import PromptLearner and TextEncoder from train.py
    import sys
    import importlib.util
    train_path = 'train.py'
    spec = importlib.util.spec_from_file_location("train", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    PromptLearner = train_module.PromptLearner
    TextEncoder = train_module.TextEncoder
    
    # Initialize prompt learner and text encoder
    prompt_learner = PromptLearner(clip_model, n_cls=len(labels)).to(device)
    text_encoder = TextEncoder(clip_model)
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    text_encoder = text_encoder.to(device)
    
    # Load checkpoint if available
    if os.path.exists(opt.checkpoint):
        checkpoint_data = torch.load(opt.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint_data, dict):
            # New format: dictionary with 'recon_net', 'prompt_learner', etc.
            if 'recon_net' in checkpoint_data:
                recon_net.load_state_dict(checkpoint_data['recon_net'])
                if 'prompt_learner' in checkpoint_data:
                    prompt_learner.load_state_dict(checkpoint_data['prompt_learner'])
                    logger.info(f"Loaded checkpoint from {opt.checkpoint} (epoch {checkpoint_data.get('epoch', 'unknown')}) - including prompt_learner")
                else:
                    logger.info(f"Loaded checkpoint from {opt.checkpoint} (epoch {checkpoint_data.get('epoch', 'unknown')}) - prompt_learner not found, using random init")
            elif 'state_dict' in checkpoint_data:
                recon_net.load_state_dict(checkpoint_data['state_dict'])
                logger.info(f"Loaded checkpoint from {opt.checkpoint}")
            else:
                # Assume it's the state dict itself
                recon_net.load_state_dict(checkpoint_data)
                logger.info(f"Loaded checkpoint from {opt.checkpoint}")
        else:
            # Old format: direct state dict
            recon_net.load_state_dict(checkpoint_data)
            logger.info(f"Loaded checkpoint from {opt.checkpoint}")
    else:
        logger.warning(f"Checkpoint {opt.checkpoint} not found. Using randomly initialized model.")
    
    # functions
    mean = np.array(preprocess.transforms[4].mean)
    std = np.array(preprocess.transforms[4].std)
    weight = np.array((0.299, 0.587, 0.114))
    gray_mean = sum(mean * weight)
    gray_std = np.sqrt(np.sum(np.power(weight,2) * np.power(std,2)))
    normal_clip = Normalize((gray_mean, ), (gray_std, ))

    # Use learned HQ prompts for text features (instead of standard CLIP)
    prompt_learner.eval()
    with torch.no_grad():
        prompts_hq = prompt_learner(hq=True)  # [n_cls, seq_len, embed_dim] - use high-quality prompts
        val_features = text_encoder(prompts_hq, None)  # [n_cls, embed_dim]
        val_features = val_features / val_features.norm(dim=-1, keepdim=True)
    logger.info(f"Using learned HQ prompts for evaluation (shape: {val_features.shape})")
    
    # -------------------- Evaluation ----------------------  
    train_start = datetime.now()
    logger.info("Start Evaluation!")
    # Metrics 
    metrics = {}
    metric_list = ['niqe','brisque','piqe']
    num_all = 0
    num_right = 0
    for metric_name in metric_list:
        metrics[metric_name] = AverageMeter()
    # visual
    for batch_idx, (spike,label,label_idx) in enumerate(tqdm(test_loader)):
        # Visual results
        spike = spike.float()
        # Dynamic voxelization: handles any T value and resamples to 50 bins
        from utils import make_voxel_224
        voxel = make_voxel_224(spike, bin_size=4, target_bins=50).to(device)  # [T,224,224] -> [50,224,224]
        spike_recon = recon_net(voxel).repeat((1,3,1,1))
        tfp = torch.mean(spike,dim = 1,keepdim = True)
        tfi = middleTFI(spike,len(spike[0])//2,len(spike[0])//4-1)
        if batch_idx % 100 == 0:
            save_img(path=f'{img_folder}/{batch_idx:04}_SpikeCLIP.png', img=normal_img(spike_recon[0,0,30:-30,10:-10]))
            save_img(path=f'{img_folder}/{batch_idx:04}_tfp.png', img=normal_img(tfp[0,0,30:-30,10:-10]))
            save_img(path=f'{img_folder}/{batch_idx:04}_tfi.png', img=normal_img(tfi[0,0,30:-30,10:-10]))
        # metric - normalize image to [0, 1] for metrics
        spike_recon_normalized = spike_recon.clone()
        spike_recon_min = spike_recon_normalized.min()
        spike_recon_max = spike_recon_normalized.max()
        if spike_recon_max > spike_recon_min:
            spike_recon_normalized = (spike_recon_normalized - spike_recon_min) / (spike_recon_max - spike_recon_min)
        spike_recon_normalized = spike_recon_normalized.clamp(0, 1)
        
        for key in metric_list:
            metrics[key].update(compute_img_metric_single(spike_recon_normalized, key, device=device))
        # cls
        spike_recon = normal_clip(spike_recon)
        image_features = clip_model.encode_image(spike_recon)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ val_features.t()
        probs = logits.softmax(dim=-1)
        index = torch.max(probs, dim=1).indices.detach().cpu()
        mask = index == label_idx
        num_right += sum(mask)
        num_all += len(mask)
    test_accuracy = 100 * num_right / num_all
    logger.info(f"Acc: {test_accuracy:.2f}")
    re_msg = ''
    for metric_name in metric_list:
        re_msg += metric_name + ": " + "{:.4f}".format(metrics[metric_name].avg) + "  "
    logger.info(re_msg)
    
    # Get checkpoint info for results tracking
    checkpoint_epoch = None
    best_val_accuracy = None
    if os.path.exists(opt.checkpoint):
        try:
            checkpoint_data = torch.load(opt.checkpoint, map_location='cpu')
            if isinstance(checkpoint_data, dict):
                checkpoint_epoch = checkpoint_data.get('epoch')
                best_val_accuracy = checkpoint_data.get('accuracy')
        except:
            pass
    
    # Save results to JSON
    try:
        from results_tracker import add_run, print_comparison
        
        # Convert metrics to float (in case they're Tensors)
        def to_float(val):
            if val is None:
                return None
            if hasattr(val, 'item'):  # Tensor
                return float(val.item())
            return float(val)
        
        add_run(
            exp_name=opt.exp_name,
            checkpoint_path=opt.checkpoint,
            best_val_accuracy=to_float(best_val_accuracy),
            best_val_epoch=checkpoint_epoch,
            test_accuracy=to_float(test_accuracy),
            niqe=to_float(metrics.get('niqe', AverageMeter()).avg) if 'niqe' in metrics else None,
            brisque=to_float(metrics.get('brisque', AverageMeter()).avg) if 'brisque' in metrics else None,
            piqe=to_float(metrics.get('piqe', AverageMeter()).avg) if 'piqe' in metrics else None,
            data_type=opt.data_type
        )
        
        # Print comparison
        print_comparison()
    except ImportError:
        logger.warning("results_tracker not available. Results not saved to JSON.")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")

