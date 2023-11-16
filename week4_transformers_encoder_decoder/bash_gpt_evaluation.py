import torch
import torch.nn as nn
import bleu_metrics


def evaluation(transformer_model,val_dl,loss_function, device):
    transformer_model.eval()
    val_correct_predictions = torch.tensor(0)
    val_correct_predictions = val_correct_predictions.to(device)
    val_total_predictions = torch.tensor(0)
    val_total_predictions = val_total_predictions.to(device)
    
    with torch.no_grad():
        for val_idx, val_batch in enumerate(val_dl):
            val_tokens = val_batch['input'].to(device)
            val_labels = val_batch['label'].to(device)
            val_output= transformer_model(val_tokens, val_tokens)
            val_model_output = val_output.view(-1, val_output.size(-1))  # Reshape to [batch_size * seq_length, num_classes]
            val_true_labels = val_labels.view(-1)  # Reshape to [batch_size * seq_length]
            val_loss = loss_function(val_model_output, val_true_labels)
            val_max_indices = torch.argmax(val_model_output, dim=1)
            val_correct_predictions += ((val_max_indices - val_true_labels)==0).sum()
            val_total_predictions += len(val_true_labels)
    val_acc = val_correct_predictions/val_total_predictions
    bleu_results = bleu_metrics.bleu_evaluate(val_model_output, val_true_labels,val_tokens.size(0))
    
    return val_acc, val_loss, bleu_results
