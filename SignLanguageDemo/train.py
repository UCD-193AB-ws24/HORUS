import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd   
import itertools
import time
from collections import defaultdict
from model import SLR
from VideoDataset import VideoDataset
from torch.nn import functional as F

import os
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

save_path = "results/"  # Directory to store models
os.makedirs(save_path, exist_ok=True)  # Create directory if not exists




learning_rate = 0.0015
n_iters = 30000
batch_size = 128
comparison_amount = 128 * 5
log_interval = 300

warm_up_iters = 6000
max_iters = 30000
weight_decay = 1e-3
final_learning_rate = learning_rate/10





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_keypoints = list(range(42)) 
selected_keypoints = selected_keypoints + [x + 42 for x in ([291, 267, 37, 61, 84, 314, 310, 13, 80, 14] + [152])]
selected_keypoints = selected_keypoints + [x + 520 for x in ([2, 5, 7, 8, 11, 12, 13, 14, 15, 16])]

flipped_selected_keypoints = list(range(21, 42)) + list(range(21)) 
flipped_selected_keypoints = flipped_selected_keypoints + [x + 42 for x in ([61, 37, 267, 291, 314, 84, 80, 13, 310, 14] + [152])]
flipped_selected_keypoints = flipped_selected_keypoints + [x + 520 for x in ([5, 2, 8, 7, 12, 11, 14, 13, 16, 15])]




def create_dataloaders(split_train_path, split_val_path, keypoints_folder, num_workers=8, prefetch_factor=8, drop_last=True):
    train_loader = DataLoader(
        VideoDataset(
            split=split_train_path,
            keypoints_path=keypoints_folder,
            video_length=64,
            selected_keypoints=selected_keypoints,
            flipped_selected_keypoints=flipped_selected_keypoints,
            augment=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last
    )

    val_loader = DataLoader(
        VideoDataset(
            split=split_val_path,
            keypoints_path=keypoints_folder,
            video_length=64,
            selected_keypoints=selected_keypoints,
            flipped_selected_keypoints=flipped_selected_keypoints,
            augment=False
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last
    )

    return train_loader, val_loader



dataset_configs = {
    "asl_citizen": {
        "train": "./Datasets/ASL_Citizen/train+test.csv",
        "val": "./Datasets/ASL_Citizen/val.csv",
        "keypoints": "./Datasets/ASL_Citizen/keypoints",
        "weight": 1.5,
    },
    "lsfb": {
        "train": "./Datasets/LSFB/train.csv",
        "val": "./Datasets/LSFB/test.csv",
        "keypoints": "./Datasets/LSFB/keypoints",
        "weight": 2
    },
    "wlasl": {
        "train": "./Datasets/WLASL/train.csv",
        "val": "./Datasets/WLASL/val.csv",
        "keypoints": "./Datasets/WLASL/keypoints",
        "weight": 0.05
    },
    "rsl": {
        "train": "./Datasets/RSL/train.csv",
        "val": "./Datasets/RSL/test.csv",
        "keypoints": "./Datasets/RSL/keypoints",
        "weight": 0.05
    },
    "autsl": {
        "train": "./Datasets/AUTSL/train+test.csv",
        "val": "./Datasets/AUTSL/val.csv",
        "keypoints": "./Datasets/AUTSL/keypoints",
        "weight": 0.5
    }
}

train_loaders = {}
val_loaders = {}

for name, cfg in dataset_configs.items():
    train_loader, val_loader = create_dataloaders(
        split_train_path=cfg["train"],
        split_val_path=cfg["val"],
        keypoints_folder=cfg["keypoints"]
    )
    train_loaders[name] = train_loader
    val_loaders[name] = val_loader





model = SLR(
    n_embd=16*64, 
    n_cls_dict={'asl_citizen':2305, 'lsfb': 4657, 'wlasl':2000, 'autsl':226, 'rsl':1001},
    n_head=16, 
    n_layer=6,
    n_keypoints=63,
    dropout=0.6, 
    max_len=64,
    bias=True
)

model = torch.compile(model)

print(f'Trainable parameters: {model.num_params()}')

criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scaler = torch.amp.GradScaler()
model.to(device)



model.load_state_dict(torch.load('./best_model_nobias_head_3.pth', map_location=torch.device(device)))



def compute_topk_accuracy(logits, labels, k=3):
    _, topk_preds = torch.topk(logits, k, dim=1)
    
    correct_topk = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    topk_correct = correct_topk.sum().item()
    return topk_correct

def get_lr(iter_num):
    if iter_num < warm_up_iters:
        return learning_rate
    elif warm_up_iters <= iter_num <= max_iters:
        return final_learning_rate + (learning_rate - final_learning_rate) * ((max_iters - iter_num) / (max_iters - warm_up_iters))
    else:
        return final_learning_rate
    


        
if __name__ == "__main__":

    
    train_iters = {name: itertools.cycle(loader) for name, loader in train_loaders.items()}
    dataset_lengths = {name: len(loader.dataset) for name, loader in train_loaders.items()}
    total_data = sum(dataset_lengths.values())
    
    best_val_loss = float('inf')

    print(dataset_lengths)
            
    train_loss_per_dataset = defaultdict(float)
    train_accuracy_per_dataset = defaultdict(float)
    train_top3_per_dataset = defaultdict(float)


    timer = time.time()
    
    for iter_num in range(1, n_iters + 1):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
        model.train()

        
        iter_loss = defaultdict(float)

        
 
        for name, train_iter in train_iters.items():
            
            optimizer.zero_grad()
            
            keypoints, valid_keypoints, labels = next(train_iter)
            keypoints = keypoints.to(device)
            valid_keypoints = valid_keypoints.to(device)
            labels = labels.to(device)
    
            with autocast(dtype=torch.bfloat16):
                logits = model.heads[name](model(keypoints, valid_keypoints))
    
            loss = criterion(logits, labels)
            weighted_loss = loss * dataset_configs[name]['weight']
            
            batch_size = labels.size(0)
            correct = (logits.argmax(dim=1) == labels).sum().item()
            top3_correct = compute_topk_accuracy(logits, labels, k=3)


            iter_loss[name] = loss.item()
            train_loss_per_dataset[name] += loss.item()/log_interval
            train_accuracy_per_dataset[name] += (100/batch_size) * correct/log_interval
            train_top3_per_dataset[name] += (100/batch_size) * top3_correct/log_interval

        
            scaler.scale(weighted_loss).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        
        #for name in iter_loss:
            #print(name, iter_loss[name])
            
    
        # === Validation + Logging ===
        if iter_num % log_interval == 0:
            model.eval()

    
            val_loss_per_dataset = {}
            val_acc_per_dataset = {}
            val_top3_per_dataset = {}
    
            with torch.no_grad():
                for name, val_loader in val_loaders.items():
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    top3_correct = 0
    
                    for keypoints, valid_keypoints, labels in val_loader:
                        keypoints = keypoints.to(device)
                        valid_keypoints = valid_keypoints.to(device)
                        labels = labels.to(device)
    
                        logits = model.heads[name](model(keypoints, valid_keypoints))
                        loss = criterion(logits, labels)
    
                        batch_size = labels.size(0)
                        val_loss += loss.item() * batch_size
                        correct += (logits.argmax(dim=1) == labels).sum().item()
                        top3_correct += compute_topk_accuracy(logits, labels, k=3)
                        total += batch_size
    
                    avg_val_loss = val_loss / total
                    val_loss_per_dataset[name] = avg_val_loss
                    val_acc_per_dataset[name] = 100 * correct / total
                    val_top3_per_dataset[name] = 100 * top3_correct / total
    
            # === Print per-dataset stats ===
            print(f"\n[Iter {iter_num}] Learning Rate: {lr:.6f} | Time: {time.time() - timer}")
            print(f"{'DATASET':<12} | Train Loss | Val Loss | Train Acc | Val Acc | Top-3 Train | Top-3 Val")
            print("-" * 80)
            for name in train_loaders.keys():
                dataset_size = dataset_lengths[name]
                train_acc = train_accuracy_per_dataset[name]
                train_top3 = train_top3_per_dataset[name]
                print(f"{name:<12} | {train_loss_per_dataset[name]:>10.4f} | {val_loss_per_dataset[name]:>8.4f} "
                      f"| {train_acc:>9.2f}% | {val_acc_per_dataset[name]:>7.2f}% "
                      f"| {train_top3:>12.2f}% | {val_top3_per_dataset[name]:>9.2f}%")
    
    
            # === Save model ===
            model_filename = f"{save_path}model_iter_{iter_num}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved at iter {iter_num}: {model_filename}")
    
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_filename = f"{save_path}best_model.pth"
                torch.save(model.state_dict(), best_model_filename)
                print(f"New best model saved at iter {iter_num}: {best_model_filename}")
                

            train_loss_per_dataset = defaultdict(float)
            train_accuracy_per_dataset = defaultdict(float)
            train_top3_per_dataset = defaultdict(float)
            
            timer = time.time()

