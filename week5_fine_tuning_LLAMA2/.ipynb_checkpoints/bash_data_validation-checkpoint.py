'''A function that defines a dataset to fine tune a large language model. 
When called, the function fetches the dataset and processes by creating prompts and tokenizing the prompts.
As such when an instance is instantiated, a tokenized dataset is created (self.ds) that is ready for training.'''
from torch.utils.data import Dataset
import torch.utils.data
import transformers as t
import datasets as d


TEMPLATE_YES_INPUT = "Below is a question that describes a task paired with further context. Write a response that appropriately completes the request.\n\n### question:\n{question}\n\n### context:\n{context}\n\n### Response:\n"
TEMPLATE_NOT_INPUT = "Below is a question that describes a task. Write a response that appropriately completes the request.\n\n### question:\n{question}\n\n### Response:\n"

class TrainDataset(Dataset):
  def __init__(self):
    self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    self.tokenizer.pad_token_id = 0
    self.tokenizer.padding_side = "right"
    self.ds = d.load_dataset("b-mc2/sql-create-context")
    self.ds = self.ds["train"].select([i for i in range(1)])
    self.ds = self.ds.map(self.prompt, remove_columns=["question", "context", "answer"], load_from_cache_file=False, num_proc=8)
    self.ds = self.ds.map(self.tokenize, remove_columns=["prompt"], load_from_cache_file=False, num_proc=8)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds[idx]

  def prompt(self, elm):
    
    TEMPLATE = TEMPLATE_NOT_INPUT if not elm["context"] else TEMPLATE_YES_INPUT
    prompt = TEMPLATE.format(question=elm["question"], context=elm["context"])
    prompt = prompt  + elm["answer"]
    return {"prompt": prompt}

  def tokenize(self, elm):
    # res = self.tokenizer(elm["prompt"])
    # res["input_ids"].append(self.tokenizer.eos_token_id)
    # res["attention_mask"].append(1)
    # res["labels"] = res["input_ids"].copy()
    res = self.tokenizer(elm["prompt"])
    
    res["labels"] = res["input_ids"].copy()
    res["labels"] = res["labels"][1:]
    res["labels"].append(self.tokenizer.eos_token_id)
    # res["input_ids"].append(self.tokenizer.bos_token_id)
    # res["attention_mask"].append(1)
    return res

  def max_seq_len(self):
    return max([len(elm["input_ids"]) for elm in self.ds])
  
  def special_tokens(self, tokenizer):
    num_added_toks = ["ALTER","DELETE", "INTO","DATABASE","DROP"]
    tokenizer.add_tokens(num_added_toks)
    return tokenizer

# ds = TrainDataset()
# dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)


# from torch.utils.data import DataLoader

# dataloader = DataLoader(
#     dataset=ds,
#     batch_size=4,  # Set batch size to 1 for simplicity
#     collate_fn=t.DataCollatorForSeq2Seq(
#         ds.tokenizer, pad_to_multiple_of=1, return_tensors="pt", padding=True
#     )
# )

# for batch in dataloader:
#     # 'batch' is a dictionary containing tokenized input and target tensors
#     # Adjust the keys based on the structure of your collate function
#     input_ids = batch["input_ids"]
#     attention_mask = batch["attention_mask"]
#     labels = batch["labels"]
#     # Print or inspect the contents of the batch
#     print("Input IDs:", input_ids)
#     print("Attention Mask:", attention_mask)
#     print("Labels:", labels)
#     break