from datasets import load_metric
import torch
import evaluate
import tokenizer_bes



def bleu_evaluate(model_output, true_labels, batch_size):
    tk = (tokenizer_bes.LangTokenizer()).load()
    sacrebleu = evaluate.load('sacrebleu')
    
    
    true_labels_chunks = torch.chunk(true_labels,batch_size)
    true_labels_chunks  = [chunk.tolist() for chunk in true_labels_chunks ]
    references = tk.decode(true_labels_chunks)

    model_predictions = torch.argmax(model_output, dim=1)
    model_predictions_chunks = torch.chunk(model_predictions,batch_size)
    model_predictions_chunks  = [chunk.tolist() for chunk in model_predictions_chunks ]
    predictions = tk.decode(model_predictions_chunks)
    results = sacrebleu.compute(predictions = predictions, references = references)
    
    return results