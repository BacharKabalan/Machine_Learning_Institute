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
  def __init__(self,train_split):
    self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    self.tokenizer.pad_token_id = 0
    self.tokenizer.padding_side = "right"
    
    self.tokenizer = self.special_tokens(self.tokenizer) #add SQL special tokens
    self.ds = d.load_dataset("b-mc2/sql-create-context")
    entire_ds = self.ds["train"]#.select([i for i in range(100)])
    dataset_len = len(entire_ds)
    dataset_len_train = int(train_split * dataset_len)
    self.ds = self.ds["train"].select([i for i in range(dataset_len_train)])
    self.ds = self.ds.map(self.prompt, remove_columns=["question", "context", "answer"], load_from_cache_file=False, num_proc=8)
    self.ds = self.ds.map(self.tokenize, remove_columns=["prompt"], load_from_cache_file=False, num_proc=8)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds[idx]

  def prompt(self, elm):
    
    TEMPLATE = TEMPLATE_NOT_INPUT if not elm["context"] else TEMPLATE_YES_INPUT
    prompt = TEMPLATE.format(question=elm["question"], context=elm["context"])
    prompt = prompt + elm["answer"]
    return {"prompt": prompt}

  def tokenize(self, elm):
    # res = self.tokenizer(elm["prompt"])
    
    # res["attention_mask"].append(1)
    # res["labels"] = res["input_ids"].copy()
    res = self.tokenizer(elm["prompt"])
    # res["input_ids"].append(self.tokenizer.eos_token_id)
    # res["attention_mask"].append(1)
    res["labels"] = res["input_ids"].copy()
    res["labels"] = res["labels"][1:]
    res["labels"].append(self.tokenizer.eos_token_id)
    # res["labels"] = res["labels"][1:]
    # res["labels"].append(self.tokenizer.eos_token_id)
    # res["input_ids"].append(self.tokenizer.bos_token_id)
    # res["attention_mask"].append(1)
    return res

  def max_seq_len(self):
    return max([len(elm["input_ids"]) for elm in self.ds])

  def special_tokens(self, tokenizer):
   
    # original_len: int = len(tokenizer)
    # num_added_toks: dict = {}
    # num_added_toks['alter_token'] = "ALTER"
    # num_added_toks['delete_token'] = "DELETE"
    # num_added_toks['into_token'] = "INTO"
    # num_added_toks['database_token'] = "DATABASE"
    # num_added_toks['drop_token'] = "DROP"
    # num_added_toks['UPDATE_token'] = "UPDATE"

    # num_added_toks['ALTER_DATABASE_token'] = "ALTER DATABASE",
    # num_added_toks['CREATE_TABLE_token'] = "CREATE TABLE",
    # num_added_toks['ALTER_TABLE_token'] = "ALTER TABLE",
    
    # num_added_toks['CREATE_INDEX_token'] = "CREATE INDEX",
    # num_added_toks['DROP_INDEX_token'] = "DROP INDEX",
    
    # num_new_tokens: int = tokenizer.add_special_tokens(num_added_toks)
    # err_msg = f"Error, not equal: {len(tokenizer)=}, {original_len + num_new_tokens=}"
    # assert len(tokenizer) == original_len + num_new_tokens, err_ms
    # assert tokenizer.SELECT_token == "SELECT"
    # return tokenizer
    # new_tokens = ["SELECT","UPDATE","DELETE", "INSERT", "INTO","CREATE","DATABASE","ALTER","TABLE","DROP","INDEX"]
    # new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    # print(new_tokens)
    # num_added_toks = [['alter_token',"ALTER"],['delete_token',"DELETE"],['into_token',"INTO"],['database_token',"DATABASE"],['drop_token',"DROP"]]
     
    sql_keywords_uppercase = ["SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "INSERT", "UPDATE", "DELETE",
    "CREATE", "ALTER", "DROP", "TABLE", "INDEX", "VIEW", "DATABASE", "DISTINCT",
    "ORDER BY", "GROUP BY", "HAVING", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
    "OUTER JOIN", "JOIN", "ON", "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
    "BETWEEN", "IN", "LIKE", "IS NULL", "IS NOT NULL", "AS", "ASC", "DESC",
    "UNION", "INTERSECT", "EXCEPT", "ALL", "ANY", "SOME", "TOP", "LIMIT", "OFFSET",
    "FETCH", "FIRST", "NEXT", "ROWS", "ONLY", "SET", "VALUES", "NULL", "TRUE", "FALSE","INTEGER", "CHAR", "VARCHAR", "TEXT", "BOOLEAN", "DATE", "TIME", "TIMESTAMP",
    "NUMERIC", "DECIMAL", "REAL", "FLOAT", "DOUBLE", "PRECISION", "BLOB", "CLOB"] 
    sql_keywords_lowercase = [keyword.lower() for keyword in sql_keywords_uppercase]
    sql_special_characeters =  [
    "*", "+", "-", "/", "=", ">", "<", "(", ")", "[", "]", "{", "}", ",", ".", ";", ":",
    "*", "(", ")", "[", "]", "{", "}", ".*", "(*)", "+", "-", "/", "=", ">", "<", "<=",
    "!=", "<>", "(", ")", "[", "]", "{", "}", ",", ".", ";", ":", "::", "::=", "||",
    "&&", "!>", "<!", "<=>", "<>", "<|", "|>", ":=", "=>", "->", "--", "/*", "*/", "//"
    ]
    num_added_toks = sql_keywords_uppercase + sql_keywords_lowercase
    num_added_toks.extend(sql_special_characeters)
    num_added_toks = set(num_added_toks) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(num_added_toks))
    # assert tokenizer.alter_token == "ALTER"
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