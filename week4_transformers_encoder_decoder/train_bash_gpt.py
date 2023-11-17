#project specific libraries
# import dataset_bes_train
# import dataset_bes_val
import dataset_bes
import tokenizer_bes
import bash_gpt_evaluation
import parameters_per_layer
import bash_gpt_inference
import bash_gpt

from torch.autograd import profiler
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
import sentencepiece as spm
import os

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

import time


seconds_in_hour = 3600
seconds_in_minute = 60

tk = (tokenizer_bes.LangTokenizer()).load()
ds = dataset_bes.LangDataset()
val_ds = dataset_bes.LangDataset()
batch_size = 4
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
checkpoint_dir = 'checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)




transformer_model = bash_gpt.GPT().cuda()
parameters_per_layer.count_parameters(transformer_model)

loss_function = torch.nn.CrossEntropyLoss()
lr = 3e-4
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr, eps=1e-9, betas=(0.9,.98))
scaler = GradScaler()


num_epochs = 1


# with profiler.profile(record_shapes=True, use_cuda=True) as prof:

user_prompt = 'Lilly and'

for epoch in range(num_epochs):
    
    transformer_model.train()
    start_time = time.time()
    for idx, batch in enumerate(dl):
#         start_time = time.time()
        enc_tokens = batch['contx'].to(device)
        dec_tokens = batch['input'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output= transformer_model(enc_tokens, dec_tokens)
            model_output = output.view(-1, output.size(-1))  # Reshape to [batch_size * seq_length, num_classes]
            true_labels = labels.view(-1)  # Reshape to [batch_size * seq_length]
            loss = loss_function(model_output, true_labels)
#             loss.backward()
#             optimizer.step()
        

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

            
#         print(time.time()-start_time)
        if idx % 1000 == 0:
            print(f"train_loss: {loss:.4f}")
            time_since_start = (time.time()-start_time)
            print(f'training time since start: {int(time_since_start/seconds_in_hour)} hours {int(time_since_start%seconds_in_hour/seconds_in_minute)} minutes')


        if idx>0 and idx%5000==0: 
            checkpoint_path = os.path.join(checkpoint_dir, f"multi_head_with_pos_encod_weights_{epoch}_{idx}.pt")
            val_acc, val_loss, bleu_metrics = bash_gpt_evaluation.evaluation(transformer_model,val_dl,loss_function, device)
            torch.save({
            'epoch': epoch,
            'model_state_dict': transformer_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'bleu_score': bleu_metrics['score']
        }, checkpoint_path)
            
            print(f"train_loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, bleu_score: {bleu_metrics['score']:4f}")
            
            
            print('*'*150 + '\n')
            print(f'User prompt -> {user_prompt}' + '\n') 
            print(f'Ba$H_GPT Story -> {bash_gpt_inference.inference(user_prompt,checkpoint_path)}' + '\n')
            print('*'*150 + '\n')

            




    time_since_start = (time.time()-start_time)
    print(f'training time since start: {int(time_since_start/seconds_in_hour)} hours {int(time_since_start%seconds_in_hour/seconds_in_minute)} minutes')
    






# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))














#     transformer_model.eval()
#     val_correct_predictions = torch.tensor(0)
#     val_correct_predictions = val_correct_predictions.to(device)
#     val_total_predictions = torch.tensor(0)
#     val_total_predictions = val_total_predictions.to(device)
#     with torch.no_grad():
#         for val_idx, val_batch in enumerate(val_dl):
#             val_tokens = val_batch['input'].to(device)
#             val_labels = val_batch['label'].to(device)
#             val_output= transformer_model(val_tokens)
#             val_model_output = val_output.view(-1, val_output.size(-1))  # Reshape to [batch_size * seq_length, num_classes]
#             val_true_labels = val_labels.view(-1)  # Reshape to [batch_size * seq_length]
#             val_loss = loss_function(val_model_output, val_true_labels)
#             val_max_indices = torch.argmax(val_model_output, dim=1)
#             val_correct_predictions += ((val_max_indices - val_true_labels)==0).sum()
#             val_total_predictions += len(val_true_labels)







#     if (epoch + 1) % checkpoint_interval == 0:
#         checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': transformer_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss
#         }, checkpoint_path)
#     print(f"Epoch {epoch+1}/{num_epochs}, val_acc: {val_acc}, val_loss: {val_loss.item()}")
    



