import transformers
import model
import bash_data_train
import bash_data_validation
import os
import torch

is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
m = model.get_model()
ds = bash_data_train.TrainDataset()
collator = transformers.DataCollatorForSeq2Seq(ds.tokenizer, pad_to_multiple_of=1, return_tensors="pt", padding=True)

validation_ds = bash_data_validation.TrainDataset()
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
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=100,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=1,
    save_steps=100,
    output_dir="./output",
    save_total_limit=3,
    ddp_find_unused_parameters=False if is_ddp else None,
  ),
)


def compute_metrics(p):
    #predictions is a numpy array with size (num_training_examples, max_sequence_length, vocab_size)
    #it gives logits. So to pass it to tokenizer.decode i need to apply softmax to it and retreive max_index. Then for each word i will get the predicted token and i should be able to visualize it
    
    
    # Extract predictions and labels from EvalPrediction
    predictions = torch.tensor(p.predictions)
    label_ids = p.label_ids
    print('************************')
    print(predictions.shape)
    print(label_ids.shape)
    print(11111111111111111111)
    
    # Create a dictionary similar to what the collator expects
    inputs = {"input_ids": predictions}

    # Perform any additional processing you need here
    # Apply softmax along the last dimension (dimension 2)
    pred_softmax_tensor = torch.nn.functional.softmax(predictions, dim=2)
    # Find the index of the maximum value along the last dimension
    pred_max_indices = torch.argmax(pred_softmax_tensor, dim=2)
    # Decode the model outputs to get the generated text
    generated_text = ds.tokenizer.decode(pred_max_indices[0], skip_special_tokens=True)
    input_text = ds.tokenizer.decode(torch.tensor(label_ids), skip_special_tokens=True)
    # Print the generated text for each batch during evaluation
    print(generated_text)
    print('******************')
    print(input_text)

    # Return the generated text as the metric
    return {"generated_text": generated_text}

# Override compute_metrics in the trainer
trainer.compute_metrics = compute_metrics

m.config.use_cache = False
trainer.train()
# trainer.evaluate()
m.save_pretrained("./weights")
