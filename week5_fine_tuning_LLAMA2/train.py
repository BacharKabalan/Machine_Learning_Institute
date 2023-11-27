import transformers
import model
import bash_data_train
import bash_data_validation
import os
import torch
import random
import evaluate


train_split = 0.999
val_split = 1-train_split
is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
m = model.get_model()
ds = bash_data_train.TrainDataset(train_split)
collator = transformers.DataCollatorForSeq2Seq(ds.tokenizer, pad_to_multiple_of=1, return_tensors="pt", padding=True)

#resize embedding layer to take into account the added tokens
m.resize_token_embeddings(len(ds.tokenizer))

validation_ds = bash_data_validation.TrainDataset(val_split)
# import torch
# import peft
# adapters_weights = torch.load("./output/checkpoint-800/adapter_model.bin")
# peft.set_peft_model_state_dict(m, adapters_weights)


trainer = transformers.Trainer(
  model=m,
  train_dataset=ds,
  eval_dataset = validation_ds,
  data_collator=collator,
  tokenizer = ds.tokenizer,
  args=transformers.TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=200,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=100,
    output_dir="./output",
    save_total_limit=5,
    ddp_find_unused_parameters=False if is_ddp else None,
  ),
)

def compute_metrics(p):
    #predictions is a numpy array with size (num_training_examples, max_sequence_length, vocab_size)
    #it gives logits. So to pass it to tokenizer.decode i need to apply softmax to it and retreive max_index. Then for each word i will get the predicted token and i should be able to visualize it
    
    
    # Extract predictions and labels from EvalPrediction
    predictions = torch.tensor(p.predictions)
    label_ids = p.label_ids
    label_ids = torch.tensor(label_ids)
    label_ids = torch.where(label_ids < 0, torch.tensor(ds.tokenizer.eos_token_id), label_ids)


    
    # Create a dictionary similar to what the collator expects
    inputs = {"input_ids": predictions}

    # Perform any additional processing you need here
    # Apply softmax along the last dimension (dimension 2)
    pred_softmax_tensor = torch.nn.functional.softmax(predictions, dim=2)
    # Find the index of the maximum value along the last dimension
    pred_max_indices = torch.argmax(pred_softmax_tensor, dim=2)
    #produce eval metrics
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')
    accuracy = evaluate.load('accuracy')
    sacrebleu = evaluate.load('sacrebleu')
    rouge = evaluate.load('rouge')
    

    
    pattern = '### Response:'
    # Decode the model outputs to get the generated text
    for record in range(pred_max_indices.size(0)):#random.sample(range(pred_max_indices.size(0)),20):
        generated_text = ds.tokenizer.decode(pred_max_indices[record], skip_special_tokens=False)
        input_text = ds.tokenizer.decode(label_ids[record], skip_special_tokens=False)
        # Print the generated text for each batch during evaluation
        # print('\n\n')
        # print('input ids \n')
        # print(pred_max_indices[record])
        # print('\n label ids \n')
        # print(label_ids[record])
        
        index_of_pattern_gen = 0#generated_text.find(pattern)
        print('Generated text: \n')
        print('\n\n')
        print('******************'*10)
        print('Generated text:\n')
        print(generated_text[index_of_pattern_gen:])
        
        index_of_pattern_input = 0#input_text.find(pattern)
        print('Input text: \n')
        print(input_text[index_of_pattern_input:])
        print('\n\n')
        precision.add_batch(predictions=pred_max_indices[record], references=label_ids[record])
        recall.add_batch(predictions=pred_max_indices[record], references=label_ids[record])
        f1.add_batch(predictions=pred_max_indices[record], references=label_ids[record])
        accuracy.add_batch(predictions=pred_max_indices[record], references=label_ids[record])
        sacrebleu.add_batch(predictions=[ds.tokenizer.decode(pred_max_indices[record].tolist())], references=[[ds.tokenizer.decode(label_ids[record].tolist())]])
        rouge.add_batch(predictions=[ds.tokenizer.decode(pred_max_indices[record].tolist())], references=[[ds.tokenizer.decode(label_ids[record].tolist())]])
    precision = precision.compute(average = 'weighted')
    recall = recall.compute(average = 'weighted')
    f1 = f1.compute(average = 'weighted')
    accuracy = accuracy.compute()
    sacrebleu = sacrebleu.compute()
    rouge = rouge.compute()
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "sacre_bleu":sacrebleu, "rouge":rouge}#{"generated_text": generated_text}

# Override compute_metrics in the trainer
trainer.compute_metrics = compute_metrics
m.config.use_cache = False
trainer.train()
# trainer.evaluate()
m.save_pretrained("./weights")