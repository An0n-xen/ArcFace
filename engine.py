import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, auc

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, _ = model(images, labels)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
            'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate_lfw(model, dataloader, device):
    """Evaluate on LFW verification task"""
    model.eval()
    
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Evaluating"):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Get embeddings
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2)
            
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy at optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_similarities)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    predictions = (all_similarities >= optimal_threshold).astype(int)
    accuracy = np.mean(predictions == all_labels)
    
    return accuracy, roc_auc, optimal_threshold