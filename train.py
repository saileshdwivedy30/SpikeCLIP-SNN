import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import clip
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
from torch.optim import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils import *
from utils import make_voxel_224  # Import helper for dynamic voxelization 
from dataset import *
# SNN model imported where needed
from torchvision.transforms import Normalize
from metrics import *
import torch.nn.functional as F
import traceback
from hq_bank import HQBank

def evaluate_accuracy(recon_net, prompt_learner, text_encoder, clip_model, val_loader, device, logger):
    """Evaluate model accuracy on validation set for early stopping"""
    recon_net.eval()
    prompt_learner.eval()
    
    # Setup CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    weight = np.array((0.299, 0.587, 0.114))
    gray_mean = sum(mean * weight)
    gray_std = np.sqrt(np.sum(np.power(weight, 2) * np.power(std, 2)))
    normal_clip = Normalize((gray_mean, ), (gray_std, ))
    
    # Get learned prompts for text features
    with torch.no_grad():
        prompts_hq = prompt_learner(hq=True)  # [n_cls, seq_len, embed_dim]
        val_features = text_encoder(prompts_hq, None)  # [n_cls, embed_dim]
        val_features = val_features / val_features.norm(dim=-1, keepdim=True)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spike, label, label_idx in val_loader:
            spike = spike.float().to(device)
            label_idx = label_idx.to(device)
            
            # Reconstruction
            voxel = make_voxel_224(spike, bin_size=4, target_bins=50).to(device)
            spike_recon = recon_net(voxel).repeat((1, 3, 1, 1))
            
            # Classification with CLIP
            spike_recon_norm = normal_clip(spike_recon)
            image_features = clip_model.encode_image(spike_recon_norm)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ val_features.t()
            probs = logits.softmax(dim=-1)
            pred_idx = torch.max(probs, dim=1).indices.detach().cpu()
            
            # Check accuracy
            correct += (pred_idx == label_idx.cpu()).sum().item()
            total += len(label_idx)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy

class TextEncoder(nn.Module):
    """CLIP text encoder wrapper for prompt learning"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts=None):
        # prompts: [n_cls, seq_len, embed_dim]
        # positional_embedding: [seq_len, embed_dim]
        # Add positional embeddings (only for the sequence length we have)
        seq_len = prompts.shape[1]
        pos_emb = self.positional_embedding[:seq_len, :].type(self.dtype)  # [seq_len, embed_dim]
        x = prompts + pos_emb.unsqueeze(0)  # [n_cls, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  # NLD -> LND: [seq_len, n_cls, embed_dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD: [n_cls, seq_len, embed_dim]
        x = self.ln_final(x).type(self.dtype)
        # Note: transformer output is [n_cls, seq_len, embed_dim]
        # Extract from EOS token position (last position) - this is where class token is
        # Shape: [n_cls, seq_len, embed_dim] -> [n_cls, embed_dim]
        x = x[:, -1, :] @ self.text_projection  # Use last position (EOS/class token)
        return x

class PromptLearner(nn.Module):
    """Learnable prompts for CLIP with dual HQ/LQ banks"""
    def __init__(self, clip_model, n_cls, n_ctx=16, class_token_position="end"):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        
        # Initialize TWO context token banks: HQ (high-quality) and LQ (low-quality)
        # Shape: [n_cls, n_ctx, ctx_dim] - each class gets its own learnable context tokens
        ctx_vectors_hq = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        ctx_vectors_lq = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_hq, std=0.02)
        nn.init.normal_(ctx_vectors_lq, std=0.02)
        self.ctx_hq = nn.Parameter(ctx_vectors_hq)
        self.ctx_lq = nn.Parameter(ctx_vectors_lq)
        
        # Keep old ctx for backward compatibility (will be removed later)
        self.ctx = self.ctx_hq  # For backward compatibility
        
        # Class token position
        self.class_token_position = class_token_position
        
        # Token embeddings - get device from clip_model
        clip_device = next(clip_model.parameters()).device
        token_embeddings = clip_model.token_embedding(clip.tokenize("a photo of a").to(clip_device)).type(dtype)
        self.register_buffer("token_prefix", token_embeddings[:, :1, :])  # SOS
        self.register_buffer("token_suffix", token_embeddings[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
    
    def forward(self, hq=True):
        """
        Args:
            hq: If True, use high-quality prompts; if False, use low-quality prompts
        Returns:
            prompts: [n_cls, seq_len, embed_dim]
        """
        ctx = self.ctx_hq if hq else self.ctx_lq  # [n_cls, n_ctx, ctx_dim]
        
        # Expand prefix and suffix to match n_cls
        prefix = self.token_prefix  # [1, 1, embed_dim]
        suffix = self.token_suffix  # [1, suffix_len, embed_dim]
        
        # Expand to match batch size (n_cls)
        prefix = prefix.expand(self.n_cls, -1, -1)  # [n_cls, 1, embed_dim]
        suffix = suffix.expand(self.n_cls, -1, -1)  # [n_cls, suffix_len, embed_dim]
        
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)  # [n_cls, seq_len, embed_dim]
        else:
            raise NotImplementedError
        
        return prompts

def freeze(model):
    """Freeze all parameters in a model"""
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(model):
    """Unfreeze all parameters in a model"""
    for p in model.parameters():
        p.requires_grad_(True)

def train_one_epoch(train_loader, recon_net, clip_model, text_encoder, prompt_learner, optimizer, 
                    scheduler, device, epoch, logger, writer, opt, current_stage=None, hq_bank=None):
    """Train for one epoch"""
    recon_net.train()
    prompt_learner.train()
    
    # Setup CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    weight = np.array((0.299, 0.587, 0.114))
    gray_mean = sum(mean * weight)
    gray_std = np.sqrt(np.sum(np.power(weight,2) * np.power(std,2)))
    normal_clip = Normalize((gray_mean, ), (gray_std, ))
    
    losses = AverageMeter()
    recon_losses = AverageMeter()
    clip_losses = AverageMeter()
    prompt_losses = AverageMeter()
    
    for batch_idx, (spike, label, label_idx) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Check prompt learner parameters AND optimizer state at start of epoch
        if batch_idx == 0:
            ctx_nan = torch.isnan(prompt_learner.ctx).any() or torch.isinf(prompt_learner.ctx).any()
            if ctx_nan:
                logger.warning(f"Epoch {epoch} start: Prompt ctx has NaN/Inf. Reinitializing...")
                with torch.no_grad():
                    nn.init.normal_(prompt_learner.ctx, std=0.02)
                if prompt_learner.ctx in optimizer.state:
                    del optimizer.state[prompt_learner.ctx]
            
            # CRITICAL: Check optimizer state for NaN (even if parameter is clean!)
            if prompt_learner.ctx in optimizer.state:
                opt_state = optimizer.state[prompt_learner.ctx]
                has_nan_state = False
                if 'exp_avg' in opt_state:
                    if torch.isnan(opt_state['exp_avg']).any() or torch.isinf(opt_state['exp_avg']).any():
                        logger.warning(f"Epoch {epoch} start: Optimizer state 'exp_avg' has NaN/Inf - RESETTING")
                        opt_state['exp_avg'].zero_()
                        has_nan_state = True
                if 'exp_avg_sq' in opt_state:
                    if torch.isnan(opt_state['exp_avg_sq']).any() or torch.isinf(opt_state['exp_avg_sq']).any():
                        logger.warning(f"Epoch {epoch} start: Optimizer state 'exp_avg_sq' has NaN/Inf - RESETTING")
                        opt_state['exp_avg_sq'].zero_()
                        has_nan_state = True
                if has_nan_state:
                    # Reset parameter too to be safe
                    with torch.no_grad():
                        nn.init.normal_(prompt_learner.ctx, std=0.02)
                    # Delete and recreate optimizer state
                    del optimizer.state[prompt_learner.ctx]
                    logger.warning(f"  Reset optimizer state completely for prompt_learner.ctx")
        
        # Text encoder is frozen - compute text features for both HQ and LQ prompts
        # Use a retry loop to handle NaN - if NaN persists, skip batch entirely
        max_retries = 3
        text_features_hq = None
        text_features_lq = None
        retry_succeeded = False
        
        for retry in range(max_retries):
            # Get learnable prompts for both HQ and LQ (require gradients)
            prompts_hq = prompt_learner(hq=True)  # [n_cls, seq_len, embed_dim]
            prompts_lq = prompt_learner(hq=False)  # [n_cls, seq_len, embed_dim]
            
            # Check for NaN in prompts and fix if needed (both banks)
            has_nan_prompts = (torch.isnan(prompts_hq).any() or torch.isinf(prompts_hq).any() or
                              torch.isnan(prompts_lq).any() or torch.isinf(prompts_lq).any())
            if has_nan_prompts:
                if batch_idx == 0 or batch_idx == len(train_loader) - 1:
                    logger.warning(f"Prompts contain NaN/Inf at epoch {epoch}, batch {batch_idx}. Reinitializing prompt learner (retry {retry+1}/{max_retries}).")
                # Reinitialize both prompt banks
                with torch.no_grad():
                    nn.init.normal_(prompt_learner.ctx_hq, std=0.02)
                    nn.init.normal_(prompt_learner.ctx_lq, std=0.02)
                # Reset optimizer state (if using Adam for prompts)
                prompts_hq = prompt_learner(hq=True)
                prompts_lq = prompt_learner(hq=False)
                continue  # Retry with new prompts
            
            # Text encoder extracts features from prompts
            text_features_hq = text_encoder(prompts_hq, None)  # [n_cls, embed_dim]
            text_features_lq = text_encoder(prompts_lq, None)  # [n_cls, embed_dim]
            
            # Check for NaN in text_features before normalization
            has_nan_text = (torch.isnan(text_features_hq).any() or torch.isinf(text_features_hq).any() or
                           torch.isnan(text_features_lq).any() or torch.isinf(text_features_lq).any())
            if has_nan_text:
                if retry == 0 and (batch_idx == 0 or batch_idx == len(train_loader) - 1):
                    logger.warning(f"Text features contain NaN/Inf at epoch {epoch}, batch {batch_idx}. Reinitializing prompts (retry {retry+1}/{max_retries}).")
                # Reinitialize prompts and recompute
                with torch.no_grad():
                    nn.init.normal_(prompt_learner.ctx_hq, std=0.02)
                    nn.init.normal_(prompt_learner.ctx_lq, std=0.02)
                prompts_hq = prompt_learner(hq=True)
                prompts_lq = prompt_learner(hq=False)
                continue  # Retry with new prompts
            
            # Normalize text features safely
            text_norm_hq = text_features_hq.norm(dim=-1, keepdim=True)
            text_norm_hq = torch.clamp(text_norm_hq, min=1e-8)
            text_features_hq = text_features_hq / text_norm_hq
            
            text_norm_lq = text_features_lq.norm(dim=-1, keepdim=True)
            text_norm_lq = torch.clamp(text_norm_lq, min=1e-8)
            text_features_lq = text_features_lq / text_norm_lq
            
            # Check again after normalization
            has_nan_normalized = (torch.isnan(text_features_hq).any() or torch.isinf(text_features_hq).any() or
                                  torch.isnan(text_features_lq).any() or torch.isinf(text_features_lq).any())
            if has_nan_normalized:
                if retry == 0 and (batch_idx == 0 or batch_idx == len(train_loader) - 1):
                    logger.warning(f"Text features contain NaN/Inf after normalization. Reinitializing prompts (retry {retry+1}/{max_retries}).")
                # Reinitialize prompts
                with torch.no_grad():
                    nn.init.normal_(prompt_learner.ctx_hq, std=0.02)
                    nn.init.normal_(prompt_learner.ctx_lq, std=0.02)
                prompts_hq = prompt_learner(hq=True)
                prompts_lq = prompt_learner(hq=False)
                continue  # Retry with new prompts
            
            # Success - no NaN
            retry_succeeded = True
            break
        
        # If still NaN after retries, skip batch entirely
        if not retry_succeeded or text_features_hq is None or text_features_lq is None:
            if batch_idx == 0 or batch_idx == len(train_loader) - 1:
                logger.warning(f"Text features still NaN/Inf after {max_retries} retries. Skipping batch {batch_idx} entirely.")
            # Reinitialize to clean state for next batch
            with torch.no_grad():
                nn.init.normal_(prompt_learner.ctx_hq, std=0.02)
                nn.init.normal_(prompt_learner.ctx_lq, std=0.02)
            # Skip the entire batch (no loss computation, no backward, no optimizer step)
            continue
        
        # Use HQ text features for class loss (backward compatibility)
        text_features = text_features_hq
        spike = spike.float().to(device)
        label_idx = label_idx.to(device)
        
        # Coarse reconstruction
        # Dynamic voxelization: handles any T value and resamples to 50 bins
        voxel = make_voxel_224(spike, bin_size=4, target_bins=50).to(device)  # [B, T, 224, 224] -> [B, 50, 224, 224]
        spike_recon = recon_net(voxel).repeat((1, 3, 1, 1))  # [B,1,224,224] -> [B,3,224,224]
        
        # --- Stage-1 coarse target (TFI) ---
        # Build a TFI target once per batch on GPU; shape to [B,1,224,224]
        # middleTFI expects [B,C,H,W] where C is time dimension
        # We have [B,T,224,224], so we can use it directly
        with torch.no_grad():
            # Process each sample in batch (middleTFI processes one batch at a time)
            tfi_list = []
            for b in range(spike.shape[0]):
                spike_b = spike[b:b+1]  # [1,T,224,224] - keep batch dim
                mid = spike_b.shape[1] // 2  # T // 2
                win = max(1, spike_b.shape[1] // 4 - 1)  # T // 4 - 1
                tfi_b = middleTFI(spike_b, mid, win)  # [1,1,224,224]
                tfi_list.append(tfi_b)
            tfi = torch.cat(tfi_list, dim=0)  # [B,1,224,224]
            tfi = tfi.clamp_min(0.0)
        
        # Replace placeholder recon loss with true TFI supervision in Stage-1
        if opt.staged and current_stage == "coarse":
            # Use L1 on grayscale channel (spike_recon is [B,3,224,224], take first channel)
            recon_loss = torch.mean(torch.abs(spike_recon[:, :1] - tfi))
        else:
            # Keep small stabilizer outside Stage-1 (or set to 0)
            recon_loss = torch.mean(torch.abs(spike_recon)) * 0.01
        
        # Check for NaN/Inf and handle gracefully (log warning, use small epsilon to maintain gradients)
        if torch.isnan(recon_loss) or torch.isinf(recon_loss):
            if batch_idx == 0 and epoch == 0:
                logger.warning(f"Recon loss is NaN/Inf at batch {batch_idx}, using small epsilon")
            recon_loss = torch.tensor(1e-6, device=device, requires_grad=True)
        
        # CLIP alignment loss
        spike_recon_norm = normal_clip(spike_recon)
        image_features = clip_model.encode_image(spike_recon_norm)  # [batch_size, embed_dim]
        
        # Normalize with epsilon to prevent division by zero
        image_norm = image_features.norm(dim=-1, keepdim=True)
        image_norm = torch.clamp(image_norm, min=1e-8)  # Prevent division by zero
        image_features = image_features / image_norm
        
        # Contrastive loss between images and all text features
        logits_per_image = image_features @ text_features.t()  # [batch_size, n_cls]
        
        # Temperature (InfoNCE-style with learnable/fixed temperature)
        tau = getattr(opt, 'temperature', 0.07)
        logits_per_image = logits_per_image / tau
        
        # Labels: the correct class indices for each image in the batch
        batch_labels = label_idx  # [batch_size]
        
        # Cross entropy loss: predict correct class for each image
        clip_loss = F.cross_entropy(logits_per_image, batch_labels)
        
        # Check for NaN/Inf and handle gracefully (log warning, use small epsilon to maintain gradients)
        if torch.isnan(clip_loss) or torch.isinf(clip_loss):
            if batch_idx == 0:
                logger.warning(f"CLIP loss is NaN/Inf at batch {batch_idx}, epoch {epoch}, using small epsilon")
                logger.warning(f"  Logits stats: min={logits_per_image.min().item():.4f}, max={logits_per_image.max().item():.4f}, mean={logits_per_image.mean().item():.4f}")
                logger.warning(f"  Image features norm: {image_features.norm().item():.4f}")
                logger.warning(f"  Text features norm: {text_features.norm().item():.4f}")
            # Use small epsilon instead of 0 to maintain gradient flow
            clip_loss = torch.tensor(1e-6, device=device, requires_grad=True)
        
        # ==== Prompt loss over HQ vs LQ (binary) in Stage-2 ====
        if opt.staged and current_stage == "prompt" and hq_bank is not None and len(hq_bank) > 0:
            # HQ images from the bank
            with torch.no_grad():
                hq_imgs = hq_bank.sample_batch(batch_size=spike.shape[0], device=device)
            # Normalize for CLIP
            hq_imgs_norm = normal_clip(hq_imgs)  # already [B,3,224,224]
            hq_img_feat = clip_model.encode_image(hq_imgs_norm)
            hq_img_feat = hq_img_feat / hq_img_feat.norm(dim=-1, keepdim=True)
            
            # LQ are our current reconstructions
            lq_img_feat = image_features.detach()
            
            # Get text (HQ/LQ) embeddings from learnable prompts
            thq_all = text_features_hq    # [n_cls,D]
            tlq_all = text_features_lq    # [n_cls,D]
            
            # Binary logits for HQ samples: compare to HQ vs LQ prompts (use class-agnostic average)
            thq_mean = thq_all.mean(0, keepdim=True)  # [1,D]
            tlq_mean = tlq_all.mean(0, keepdim=True)  # [1,D]
            
            hq_logits = torch.stack([(hq_img_feat * thq_mean).sum(-1),
                                     (hq_img_feat * tlq_mean).sum(-1)], dim=1)  # [B,2]
            lq_logits = torch.stack([(lq_img_feat * thq_mean).sum(-1),
                                     (lq_img_feat * tlq_mean).sum(-1)], dim=1)  # [B,2]
            
            # HQ should be class 0; LQ should be class 1
            prompt_loss = F.cross_entropy(hq_logits, torch.zeros(hq_logits.size(0), dtype=torch.long, device=device)) \
                        + F.cross_entropy(lq_logits, torch.ones(lq_logits.size(0), dtype=torch.long, device=device))
        else:
            # Fallback: encourage image features to align with HQ prompts over LQ prompts
            # Gather class-specific text embeddings for the batch
            thq_y = text_features_hq[label_idx]   # [B, D] - HQ embeddings for true classes
            tlq_y = text_features_lq[label_idx]   # [B, D] - LQ embeddings for true classes
            
            # Compute similarity scores: image_features vs HQ and LQ
            score_hq = (image_features * thq_y).sum(dim=-1)  # [B] - score w.r.t. HQ
            score_lq = (image_features * tlq_y).sum(dim=-1)  # [B] - score w.r.t. LQ
            
            # Stack into [B, 2] where class 0 = HQ, class 1 = LQ
            two_logits = torch.stack([score_hq, score_lq], dim=1)  # [B, 2]
            
            # Cross-entropy loss: encourage HQ (class 0) to be preferred
            prompt_loss = F.cross_entropy(two_logits, torch.zeros_like(label_idx))  # All labels are 0 (HQ)
        
        # Check for NaN/Inf in prompt loss
        if torch.isnan(prompt_loss) or torch.isinf(prompt_loss):
            if batch_idx == 0:
                logger.warning(f"Prompt loss is NaN/Inf at batch {batch_idx}, epoch {epoch}, using small epsilon")
            prompt_loss = torch.tensor(1e-6, device=device, requires_grad=True)
        
        # Total loss - stage-dependent
        if opt.staged and current_stage is not None:
            if current_stage == "coarse":
                # Stage 1: Only reconstruction loss
                total_loss = opt.recon_weight * recon_loss
            elif current_stage == "prompt":
                # Stage 2: Prompt loss + class loss (no reconstruction)
                total_loss = opt.clip_weight * clip_loss + opt.prompt_weight * prompt_loss
            else:  # fine
                # Stage 3: All losses
                total_loss = (opt.recon_weight * recon_loss + 
                             opt.clip_weight * clip_loss + 
                             opt.prompt_weight * prompt_loss)
        else:
            # Single-stage: all losses
            total_loss = (opt.recon_weight * recon_loss + 
                         opt.clip_weight * clip_loss + 
                         opt.prompt_weight * prompt_loss)
        
        # Check for NaN in loss before backward
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"Epoch {epoch}, batch {batch_idx}: Total loss is NaN/Inf before backward! Recon={recon_loss.item():.6e}, CLIP={clip_loss.item():.6e}")
            # Skip this batch
            continue
        
        # Backward with NaN prevention
        optimizer.zero_grad()
        
        # Check parameters before backward (critical check - skip if NaN)
        ctx_nan_before_backward = torch.isnan(prompt_learner.ctx).any() or torch.isinf(prompt_learner.ctx).any()
        if ctx_nan_before_backward:
            logger.warning(f"Epoch {epoch}, batch {batch_idx}: Prompt ctx has NaN/Inf BEFORE backward() - skipping backward")
            # Reinitialize and skip this batch
            with torch.no_grad():
                nn.init.normal_(prompt_learner.ctx, std=0.02)
            if prompt_learner.ctx in optimizer.state:
                del optimizer.state[prompt_learner.ctx]
            continue  # Skip this batch
        
        # Perform backward pass
        try:
            total_loss.backward()
        except RuntimeError as e:
            if "NaN" in str(e) or "Inf" in str(e):
                logger.warning(f"Epoch {epoch}, batch {batch_idx}: RuntimeError during backward: {e}")
                # Reinitialize and skip
                with torch.no_grad():
                    nn.init.normal_(prompt_learner.ctx, std=0.02)
                if prompt_learner.ctx in optimizer.state:
                    del optimizer.state[prompt_learner.ctx]
                continue
            else:
                raise
        
        # Check parameters after backward (ALL batches - critical!)
        ctx_nan_after_backward = torch.isnan(prompt_learner.ctx).any() or torch.isinf(prompt_learner.ctx).any()
        if ctx_nan_after_backward:
            logger.warning(f"Epoch {epoch}, batch {batch_idx}: Prompt ctx has NaN/Inf AFTER backward() - reinitializing and SKIPPING optimizer.step()")
            # Reinitialize immediately
            with torch.no_grad():
                nn.init.normal_(prompt_learner.ctx, std=0.02)
            # Zero gradients
            if prompt_learner.ctx.grad is not None:
                prompt_learner.ctx.grad.zero_()
            # Reset optimizer state
            if prompt_learner.ctx in optimizer.state:
                del optimizer.state[prompt_learner.ctx]
            # CRITICAL: Skip optimizer step for this batch - NO LOGGING AFTER THIS!
            continue  # Skip optimizer step - this prevents NaN from being applied!
        
        # Also check gradients for prompt_learner.ctx specifically (before gradient clipping)
        if prompt_learner.ctx.grad is not None:
            ctx_grad_nan = torch.isnan(prompt_learner.ctx.grad).any() or torch.isinf(prompt_learner.ctx.grad).any()
            if ctx_grad_nan:
                logger.warning(f"Epoch {epoch}, batch {batch_idx}: Prompt ctx GRADIENT has NaN/Inf AFTER backward() - reinitializing and SKIPPING optimizer.step()")
                # Reinitialize immediately
                with torch.no_grad():
                    nn.init.normal_(prompt_learner.ctx, std=0.02)
                prompt_learner.ctx.grad.zero_()
                # Reset optimizer state
                if prompt_learner.ctx in optimizer.state:
                    del optimizer.state[prompt_learner.ctx]
                # Skip optimizer step
                continue
        
        # Check for NaN gradients BEFORE clipping (critical check!)
        has_nan_grad = False
        for param in list(recon_net.parameters()) + list(prompt_learner.parameters()):
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                param_name = "prompt_learner.ctx" if param is prompt_learner.ctx else "other"
                logger.warning(f"NaN/Inf gradients detected at epoch {epoch}, batch {batch_idx}, param={param_name}. Reinitializing and skipping optimizer.step().")
                if param is prompt_learner.ctx:
                    # Reinitialize parameter if it's the prompt learner
                    with torch.no_grad():
                        nn.init.normal_(prompt_learner.ctx, std=0.02)
                param.grad.zero_()
                # Reset optimizer state
                if param is prompt_learner.ctx and param in optimizer.state:
                    del optimizer.state[param]
                # Skip optimizer step for this batch
                break  # Exit loop, we'll skip optimizer step
        
        # If NaN gradients detected, skip optimizer step
        if has_nan_grad:
            if batch_idx == 0 or batch_idx == len(train_loader) - 1:
                logger.warning(f"  Skipping optimizer.step() due to NaN gradients")
            continue
        
        # Only clip gradients if no NaN detected
        recon_grad_norm = torch.nn.utils.clip_grad_norm_(recon_net.parameters(), max_norm=1.0)
        prompt_grad_norm = torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), max_norm=1.0)
        
        # Check gradient norms (should be finite) - ALL batches!
        if not (torch.isfinite(torch.tensor(recon_grad_norm, dtype=torch.float32)) and torch.isfinite(torch.tensor(prompt_grad_norm, dtype=torch.float32))):
            logger.warning(f"Epoch {epoch}, batch {batch_idx}: Gradient norms are not finite: Recon={recon_grad_norm:.4f}, Prompt={prompt_grad_norm:.4f}. Skipping optimizer.step().")
            # Reinitialize to be safe
            with torch.no_grad():
                nn.init.normal_(prompt_learner.ctx, std=0.02)
            if prompt_learner.ctx in optimizer.state:
                del optimizer.state[prompt_learner.ctx]
            continue
        
        # Check parameters before optimizer step (all batches) - check both banks
        ctx_hq_nan_before_step = torch.isnan(prompt_learner.ctx_hq).any() or torch.isinf(prompt_learner.ctx_hq).any()
        ctx_lq_nan_before_step = torch.isnan(prompt_learner.ctx_lq).any() or torch.isinf(prompt_learner.ctx_lq).any()
        if ctx_hq_nan_before_step or ctx_lq_nan_before_step:
            if batch_idx == len(train_loader) - 1:
                logger.warning(f"Epoch {epoch}, last batch: Prompt ctx_hq or ctx_lq has NaN/Inf BEFORE optimizer.step() - reinitializing and skipping step")
            # Reinitialize before optimizer step
            with torch.no_grad():
                nn.init.normal_(prompt_learner.ctx_hq, std=0.02)
                nn.init.normal_(prompt_learner.ctx_lq, std=0.02)
            # Skip optimizer step for this batch
            continue
        
        # Final check before optimizer.step() - double safety!
        ctx_hq_nan_final = torch.isnan(prompt_learner.ctx_hq).any() or torch.isinf(prompt_learner.ctx_hq).any()
        ctx_lq_nan_final = torch.isnan(prompt_learner.ctx_lq).any() or torch.isinf(prompt_learner.ctx_lq).any()
        if ctx_hq_nan_final or ctx_lq_nan_final:
            logger.warning(f"Epoch {epoch}, batch {batch_idx}: Prompt ctx_hq or ctx_lq has NaN/Inf in FINAL check before optimizer.step() - skipping!")
            with torch.no_grad():
                nn.init.normal_(prompt_learner.ctx_hq, std=0.02)
                nn.init.normal_(prompt_learner.ctx_lq, std=0.02)
            continue
        
        # CRITICAL FIX: Adam optimizer keeps introducing NaN via momentum buffers
        # Solution: Use manual SGD update for prompt_learner.ctx_hq and ctx_lq to avoid momentum issues
        # Update recon_net with optimizer as normal
        optimizer.step()
        
        # For prompt_learner.ctx_hq and ctx_lq, use manual SGD update (no momentum) to avoid NaN
        # Use a higher learning rate for prompt learning to ensure it learns effectively
        lr = optimizer.param_groups[0]['lr']
        prompt_lr = lr * 10.0  # Higher LR for prompt learning
        
        # Update both HQ and LQ prompt banks
        for ctx_param, ctx_name in [(prompt_learner.ctx_hq, 'ctx_hq'), (prompt_learner.ctx_lq, 'ctx_lq')]:
            if ctx_param.grad is not None:
                # Check gradient is valid
                grad_valid = not (torch.isnan(ctx_param.grad).any() or torch.isinf(ctx_param.grad).any())
                if grad_valid:
                    # Manual SGD update: param = param - lr * grad
                    with torch.no_grad():
                        # Clip gradient before update to prevent instability
                        grad_norm = ctx_param.grad.norm()
                        if grad_norm > 1.0:
                            ctx_param.grad = ctx_param.grad / grad_norm * 1.0
                        # Log gradient stats occasionally for debugging
                        if batch_idx == 0 and epoch % 2 == 0:
                            logger.info(f"Epoch {epoch}, batch {batch_idx}: {ctx_name} grad norm={grad_norm:.6f}, LR={prompt_lr:.6e}")
                        ctx_param.data -= prompt_lr * ctx_param.grad
                        # Verify update didn't introduce NaN
                        if torch.isnan(ctx_param).any() or torch.isinf(ctx_param).any():
                            logger.warning(f"Epoch {epoch}, batch {batch_idx}: Manual SGD update for {ctx_name} introduced NaN - reverting")
                            nn.init.normal_(ctx_param, std=0.02)
                    # Zero gradient after manual update
                    ctx_param.grad.zero_()
                else:
                    logger.warning(f"Epoch {epoch}, batch {batch_idx}: {ctx_name} gradient has NaN/Inf - skipping update")
                    ctx_param.grad.zero_()
        
        # Check parameters after update (sanity check)
        ctx_nan_after_step = torch.isnan(prompt_learner.ctx).any() or torch.isinf(prompt_learner.ctx).any()
        if ctx_nan_after_step:
            logger.warning(f"Epoch {epoch}, batch {batch_idx}: Prompt ctx has NaN/Inf after update - reinitializing")
            with torch.no_grad():
                nn.init.normal_(prompt_learner.ctx, std=0.02)
            # Delete optimizer state if it exists
            if prompt_learner.ctx in optimizer.state:
                del optimizer.state[prompt_learner.ctx]
        
        # Log raw values before any clamping
        total_loss_val = total_loss.item()
        recon_loss_val = recon_loss.item()
        clip_loss_val = clip_loss.item()
        
        # Check if losses are suspiciously small (might indicate numerical issues)
        if batch_idx == 0 and epoch > 0:
            if total_loss_val < 1e-6 and recon_loss_val < 1e-6 and clip_loss_val < 1e-6:
                logger.warning(f"Very small losses at epoch {epoch}, batch {batch_idx}: Loss={total_loss_val:.6e}, Recon={recon_loss_val:.6e}, CLIP={clip_loss_val:.6e}")
                logger.warning(f"  Spike recon stats: min={spike_recon.min().item():.4f}, max={spike_recon.max().item():.4f}, mean={spike_recon.mean().item():.4f}")
                logger.warning(f"  Image features stats: min={image_features.min().item():.4f}, max={image_features.max().item():.4f}")
                logger.warning(f"  Logits stats: min={logits_per_image.min().item():.4f}, max={logits_per_image.max().item():.4f}")
        
        # Update loss meters
        prompt_loss_val = prompt_loss.item() if hasattr(prompt_loss, 'item') else prompt_loss
        losses.update(total_loss_val)
        recon_losses.update(recon_loss_val)
        clip_losses.update(clip_loss_val)
        prompt_losses.update(prompt_loss_val)
        
        if batch_idx % 100 == 0:
            stage_info = f", Stage={current_stage}" if current_stage else ""
            logger.info(f"Batch {batch_idx}: Loss={total_loss_val:.6f}, Recon={recon_loss_val:.6f}, CLIP={clip_loss_val:.6f}, Prompt={prompt_loss_val:.6f}{stage_info}")
            writer.add_scalar('Train/Loss', total_loss_val, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Train/ReconLoss', recon_loss_val, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Train/CLIPLoss', clip_loss_val, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Train/PromptLoss', prompt_loss_val, epoch * len(train_loader) + batch_idx)
    
    # Check prompt learner parameters before scheduler step
    if batch_idx == len(train_loader) - 1:  # Last batch of epoch
        ctx_nan_before = torch.isnan(prompt_learner.ctx).any() or torch.isinf(prompt_learner.ctx).any()
        ctx_stats_before = (prompt_learner.ctx.min().item(), prompt_learner.ctx.max().item(), prompt_learner.ctx.mean().item())
        if ctx_nan_before:
            logger.warning(f"Epoch {epoch} end: Prompt ctx has NaN/Inf before scheduler step")
    
    if scheduler is not None:
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or (batch_idx == len(train_loader) - 1):
            logger.info(f"Epoch {epoch}: LR changed from {current_lr:.6e} to {new_lr:.6e}")
    
    # Check prompt learner parameters after scheduler step
    if batch_idx == len(train_loader) - 1:  # Last batch of epoch
        ctx_nan_after = torch.isnan(prompt_learner.ctx).any() or torch.isinf(prompt_learner.ctx).any()
        ctx_stats_after = (prompt_learner.ctx.min().item(), prompt_learner.ctx.max().item(), prompt_learner.ctx.mean().item())
        if ctx_nan_after:
            logger.warning(f"Epoch {epoch} end: Prompt ctx has NaN/Inf after scheduler step")
            logger.warning(f"  Before: min={ctx_stats_before[0]:.4f}, max={ctx_stats_before[1]:.4f}, mean={ctx_stats_before[2]:.4f}")
            logger.warning(f"  After: min={ctx_stats_after[0]:.4f}, max={ctx_stats_after[1]:.4f}, mean={ctx_stats_after[2]:.4f}")
    
    logger.info(f"Epoch {epoch} - Loss: {losses.avg:.6f}, Recon: {recon_losses.avg:.6f}, CLIP: {clip_losses.avg:.6f}, Prompt: {prompt_losses.avg:.6f}")
    # Warn if average losses are suspiciously small
    if losses.avg < 1e-5:
        logger.warning(f"Epoch {epoch}: Average loss is very small ({losses.avg:.6e}), might indicate training issues")
    return losses.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='spikeclip_train')
    parser.add_argument('--data_type', type=str, default='CALTECH')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=25, help='Total epochs (default: 25 to match paper)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--clip_weight', type=float, default=0.1)
    parser.add_argument('--prompt_weight', type=float, default=0.75)
    parser.add_argument('--staged', action='store_true', help='Use three-stage training (coarse -> prompt -> fine). Default: True (paper-compliant)')
    parser.add_argument('--no-staged', dest='staged', action='store_false', help='Disable three-stage training (use single-stage)')
    parser.set_defaults(staged=True)  # Default to True (paper-compliant)
    parser.add_argument('--epochs_coarse', type=int, default=5, help='Epochs for Stage 1 (coarse reconstruction)')
    parser.add_argument('--epochs_prompt', type=int, default=1, help='Epochs for Stage 2 (prompt learning)')
    parser.add_argument('--epochs_fine', type=int, default=19, help='Epochs for Stage 3 (refinement)')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE-style CLIP loss')
    parser.add_argument('--hq_bank_path', type=str, default='data/HQ-IMAGES', help='Path to HQ image bank for Stage-2')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--early_stop', action='store_true', default=False, help='Enable early stopping based on validation accuracy')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stopping (only if --early_stop is enabled)')
    
    opt = parser.parse_args()
    
    # Labels
    if opt.data_type == 'CALTECH':
        labels = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background']
        opt.base_folder = 'data/U-CALTECH'
    elif opt.data_type == 'CIFAR':
        labels = ["frog","horse","dog","truck","airplane","automobile","bird","ship","cat","deer"]
        opt.base_folder = 'data/U-CIFAR'
    
    opt.labels = labels
    
    # Setup directories
    exp_dir = f"exp/{opt.exp_name}"
    ckpt_dir = f"{exp_dir}/ckpts"
    img_dir = f"{exp_dir}/imgs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    set_random_seed(opt.seed)
    save_opt(opt, f"{exp_dir}/opt.txt")
    log_file = f"{exp_dir}/train.log"
    logger = setup_logging(log_file)
    
    if os.path.exists(f'{exp_dir}/tensorboard'):
        shutil.rmtree(f'{exp_dir}/tensorboard')
    writer = SummaryWriter(f'{exp_dir}/tensorboard')
    logger.info(opt)
    
    # Data loaders
    # Note: If early_stop is enabled, train/val split will be done below
    train_dataset_full = SpikeData(opt.base_folder, labels, stage='train')
    train_loader = None  # Will be set below
    
    # Validation loader for early stopping (if enabled)
    # Use subset of training data as validation to avoid using test set
    val_loader = None
    if opt.early_stop:
        # Split training data into train/val (80/20 split)
        train_size = int(0.8 * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size
        
        # Use random split for validation
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset_full, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(opt.seed)
        )
        
        # Recreate train_loader with split dataset
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True, 
            num_workers=2, pin_memory=True
        )
        
        # Create validation loader
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, 
            num_workers=1, pin_memory=False  # Smaller batch for evaluation
        )
        logger.info(f"Early stopping enabled: patience={opt.patience} epochs")
        logger.info(f"Split training data: {train_size} train, {val_size} validation")
        logger.info(f"Will evaluate on {val_size} validation samples each epoch in Stage 3 (fine-tuning) only")
        logger.info(f"Early stopping active only in Stage 3 - accuracy monitoring meaningful when classification learning begins")
    else:
        # No early stopping: use full training set
        train_loader = torch.utils.data.DataLoader(
            train_dataset_full, batch_size=opt.batch_size, shuffle=True, 
            num_workers=2, pin_memory=True  # Reduced from 4 to 2 to avoid memory issues
        )
    
    # Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    # Reconstruction network (SNN)
    from models import SNN_LRN_Wrapper
    recon_net = SNN_LRN_Wrapper(inDim=50, outDim=1, num_steps=50).to(device)
    logger.info("Using SNN (Spiking Neural Network) model")
    
    # Prompt learner (needs number of classes)
    prompt_learner = PromptLearner(clip_model, n_cls=len(labels)).to(device)
    text_encoder = TextEncoder(clip_model)
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    text_encoder = text_encoder.to(device)
    
    # HQ Bank for Stage-2 prompt learning (optional - will work even if empty)
    hq_bank = None
    hq_bank_path = getattr(opt, 'hq_bank_path', 'data/HQ-IMAGES')
    if os.path.exists(hq_bank_path):
        try:
            hq_bank = HQBank(root=hq_bank_path, image_size=224)
            if len(hq_bank) == 0:
                logger.warning(f"HQ bank directory exists but is empty: {hq_bank_path}")
                hq_bank = None
            else:
                logger.info(f"HQ bank loaded: {len(hq_bank)} images available")
        except Exception as e:
            logger.warning(f"Failed to load HQ bank from {hq_bank_path}: {e}")
            hq_bank = None
    else:
        logger.info(f"HQ bank directory not found: {hq_bank_path}. Stage-2 will use alternative prompt loss.")
    
    # Optimizer - only for recon_net (prompt_learner will use manual SGD)
    # This avoids Adam's momentum buffers causing NaN issues
    optimizer = Adam(
        list(recon_net.parameters()),  # Only recon_net uses Adam
        lr=opt.lr
    )
    # Note: prompt_learner.ctx will be updated manually with SGD to avoid momentum NaN issues
    
    # Compute stage boundaries if using staged training
    if opt.staged:
        b0 = 0
        b1 = opt.epochs_coarse
        b2 = b1 + opt.epochs_prompt
        total_epochs = b2 + opt.epochs_fine
        logger.info(f"Staged training: Stage 1 (coarse) epochs 0-{b1-1}, Stage 2 (prompt) epochs {b1}-{b2-1}, Stage 3 (fine) epochs {b2}-{total_epochs-1}")
    else:
        total_epochs = opt.epochs
        b1 = b2 = None  # Not used in non-staged mode
    
    # Initial scheduler (will be updated if resuming)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    
    # Stage management function
    current_stage = [None]  # Use list to allow modification in nested function
    scheduler_ref = [scheduler]  # Use list to allow modification in nested function
    
    def set_stage(epoch):
        if opt.staged:
            if epoch < b1:
                stage = "coarse"
            elif epoch < b2:
                stage = "prompt"
            else:
                stage = "fine"
        else:
            stage = None
        
        if stage == current_stage[0]:
            return stage
        
        current_stage[0] = stage
        
        if opt.staged:
            if stage == "coarse":
                unfreeze(recon_net)
                freeze(prompt_learner)
                opt.clip_weight = 0.0  # No CLIP/class or prompt loss
                logger.info(f"==> Stage 1 (coarse): Training recon_net only, epochs {epoch}-{b1-1}")
            elif stage == "prompt":
                freeze(recon_net)
                unfreeze(prompt_learner)
                opt.clip_weight = 1.0  # Keep class CE on
                logger.info(f"==> Stage 2 (prompt): Training prompts only, epochs {epoch}-{b2-1}")
            elif stage == "fine":
                unfreeze(recon_net)
                unfreeze(prompt_learner)
                opt.clip_weight = 1.0
                logger.info(f"==> Stage 3 (fine): Joint training, epochs {epoch}-{total_epochs-1}")
            
            # Restart LR schedule at each stage
            for pg in optimizer.param_groups:
                pg['lr'] = opt.lr
            
            # Create new scheduler for current stage
            if stage == "coarse":
                scheduler_ref[0] = CosineAnnealingLR(optimizer, T_max=b1)
            elif stage == "prompt":
                scheduler_ref[0] = CosineAnnealingLR(optimizer, T_max=(b2 - b1))
            else:  # fine
                scheduler_ref[0] = CosineAnnealingLR(optimizer, T_max=(total_epochs - b2))
        
        return stage
    
    # Resume from checkpoint
    start_epoch = 0
    if opt.resume and os.path.exists(opt.resume):
        checkpoint = torch.load(opt.resume)
        recon_net.load_state_dict(checkpoint['recon_net'])
        prompt_learner.load_state_dict(checkpoint['prompt_learner'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed from epoch {start_epoch}")
        
        # CRITICAL: Update total_epochs if --epochs is different from staged schedule
        # This allows resuming with more epochs (e.g., resume from 25, train to 50)
        if opt.epochs > total_epochs:
            # User wants more epochs than the staged schedule
            if opt.staged:
                # Extend Stage 3 (fine-tuning) to reach opt.epochs
                total_epochs = opt.epochs
                logger.info(f"Extended training to {total_epochs} epochs (Stage 3 extended to epochs {b2}-{total_epochs-1})")
            else:
                total_epochs = opt.epochs
                logger.info(f"Updated total epochs to {total_epochs}")
        elif opt.epochs < total_epochs:
            # User wants fewer epochs
            total_epochs = opt.epochs
            logger.info(f"Updated total epochs to {total_epochs}")
        
        # CRITICAL: Reset learning rate when resuming (scheduler may have finished)
        # Set LR back to initial value for continued training
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr
        logger.info(f"Reset learning rate to {opt.lr} for continued training")
        
        # Create new scheduler for remaining epochs
        remaining_epochs = total_epochs - start_epoch
        if remaining_epochs > 0:
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            logger.info(f"Created new scheduler for {remaining_epochs} remaining epochs")
        else:
            logger.warning(f"No remaining epochs! start_epoch={start_epoch}, total_epochs={total_epochs}")
    
    # Training loop with error handling
    logger.info("Starting training...")
    
    # Early stopping variables
    best_accuracy = 0.0
    patience_counter = 0
    best_epoch = 0
    
    try:
        for epoch in range(start_epoch, total_epochs):
            try:
                # Set stage (freeze/unfreeze, adjust losses)
                stage = set_stage(epoch)
                
                train_loss = train_one_epoch(
                    train_loader, recon_net, clip_model, text_encoder, prompt_learner,
                    optimizer, scheduler_ref[0], device, epoch, logger, writer, opt, current_stage=stage, hq_bank=hq_bank
                )
                
                # Early stopping evaluation (if enabled)
                # Only check validation accuracy in Stage 3 (fine-tuning) when classification matters
                # Stage 1/2 don't need early stopping (accuracy won't improve during reconstruction/prompt learning)
                current_accuracy = 0.0
                if opt.early_stop and val_loader is not None and stage == "fine":
                    logger.info(f"Evaluating on validation set for early stopping...")
                    current_accuracy = evaluate_accuracy(
                        recon_net, prompt_learner, text_encoder, clip_model, 
                        val_loader, device, logger
                    )
                    logger.info(f"Epoch {epoch+1}: Validation Accuracy = {current_accuracy:.2f}%")
                    writer.add_scalar('Accuracy/Validation', current_accuracy, epoch+1)
                    
                    # Check if this is the best model (only in Stage 3)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_epoch = epoch + 1
                        patience_counter = 0
                        
                        # Save best checkpoint
                        best_checkpoint = {
                            'epoch': epoch + 1,
                            'recon_net': recon_net.state_dict(),
                            'prompt_learner': prompt_learner.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'accuracy': current_accuracy,
                        }
                        torch.save(best_checkpoint, f"{ckpt_dir}/checkpoint_best.pth")
                        logger.info(f"✅ New best accuracy: {current_accuracy:.2f}% (epoch {epoch+1}) - saved as checkpoint_best.pth")
                    else:
                        patience_counter += 1
                        logger.info(f"No improvement for {patience_counter}/{opt.patience} epochs (best: {best_accuracy:.2f}% at epoch {best_epoch})")
                        
                        # Early stopping check
                        if patience_counter >= opt.patience:
                            logger.info(f"⏹️  Early stopping triggered! No improvement for {opt.patience} epochs.")
                            logger.info(f"Best model: Epoch {best_epoch} with accuracy {best_accuracy:.2f}%")
                            logger.info(f"Restored best checkpoint from epoch {best_epoch}")
                            
                            # Load best checkpoint
                            best_checkpoint = torch.load(f"{ckpt_dir}/checkpoint_best.pth", map_location=device)
                            recon_net.load_state_dict(best_checkpoint['recon_net'])
                            prompt_learner.load_state_dict(best_checkpoint['prompt_learner'])
                            logger.info(f"Training stopped early. Best checkpoint loaded.")
                            break
                elif opt.early_stop and stage != "fine":
                    # Skip early stopping check in Stage 1/2 (accuracy not meaningful yet)
                    logger.debug(f"Skipping early stopping check in {stage} stage (only active in fine-tuning stage)")
                
                # Save checkpoint (regular intervals)
                if (epoch + 1) % 5 == 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'recon_net': recon_net.state_dict(),
                        'prompt_learner': prompt_learner.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    if opt.early_stop:
                        checkpoint['accuracy'] = current_accuracy
                    torch.save(checkpoint, f"{ckpt_dir}/checkpoint_epoch_{epoch+1}.pth")
                    logger.info(f"Saved checkpoint at epoch {epoch+1}")
            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Save checkpoint even on error
                checkpoint = {
                    'epoch': epoch,
                    'recon_net': recon_net.state_dict(),
                    'prompt_learner': prompt_learner.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, f"{ckpt_dir}/checkpoint_epoch_{epoch}_error.pth")
                logger.info(f"Saved error checkpoint at epoch {epoch}")
                raise  # Re-raise to stop training
        
        if opt.early_stop and patience_counter < opt.patience:
            logger.info("Training completed!")
            if best_accuracy > 0:
                logger.info(f"Best model: Epoch {best_epoch} with accuracy {best_accuracy:.2f}%")
        else:
            logger.info("Training completed!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error during training: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        writer.close()

