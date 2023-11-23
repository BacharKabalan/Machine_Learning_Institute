#%%
import transformers as t
import torch
import peft
import bash_data_train
import time
#%%
ds = bash_data_train.TrainDataset()
tokenizer = ds.tokenizer
# tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
base_model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
print(len(tokenizer))
# tokenizer.pad_token_id = 0
#%%
checkpoint_path = "./output/checkpoint-4500"
config = peft.PeftConfig.from_pretrained(checkpoint_path)
model = peft.PeftModel.from_pretrained(base_model, checkpoint_path)
# peft.set_peft_model_state_dict(model,"./output/checkpoint-4900/adapter_config.json")
#%%
model.resize_token_embeddings(len(tokenizer))


########### ADD Template in a file
TEMPLATE_YES_INPUT = "Below is a question that describes a task paired with further context. Write a response that appropriately completes the request.\n\n### question:\n{question}\n\n### context:\n{context}\n\n### Response:\n"
QUESTION = "How many heads of the departments are older than 56 ?"
CONTEXT = "CREATE TABLE head (age INTEGER)"
prompt = TEMPLATE_YES_INPUT.format(question=QUESTION, context=CONTEXT)
#%%

pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)

print(pipe(prompt)[0]['generated_text'])
# print("pipe(prompt)", pipe(prompt))