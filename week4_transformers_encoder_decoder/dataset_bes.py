import torch
import pandas
import tokenizer_bes


class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    f = open('./names.txt', 'r')
    self.names = f.read().split('\n')
    self.tknz  = tokenizer_bes.Tokenizer()
    f.close()

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name  = self.names[idx]
    input = [self.tknz.stoi['<sos>']] + self.tknz.encode(name)
    label = (input[1:]) + [self.tknz.stoi['<eos>']]
    masks = [1] * len(input)
    return {
      'plain': name,
      'input': torch.tensor(input),
      'label': torch.tensor(label),
      'masks': torch.tensor(masks),
    }

  # The input batch is a list of tensors with different
  # lengths. We use pad_sequence to pad the tensors with
  # 0s so that they all have the same length.
  def collate_fn(self, batch):
    input_pad = torch.nn.utils.rnn.pad_sequence([item['input'] for item in batch], batch_first=True, padding_value=0)
    label_pad = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=0)
    masks_pad = torch.nn.utils.rnn.pad_sequence([item['masks'] for item in batch], batch_first=True, padding_value=0)

    return {
      'plain': [item['plain'] for item in batch],
      'input': input_pad,
      'label': label_pad,
      'masks': masks_pad,
    }


class LangDataset(torch.utils.data.Dataset):
  def __init__(self, data_split):
    self.column_names = ['id_eng', 'eng', 'id_ita', 'ita']
    self.df = pandas.read_csv('./eng_ita.tsv', delimiter='\t', encoding='utf-8', on_bad_lines='skip', header=None, names=self.column_names)
    if data_split>0.7:
      train_index = int(data_split*len(self.df))
      self.df = self.df[:train_index]
    else:
      validation_index = len(self.df) - int(len(self.df)*0.2)
      self.df = self.df[validation_index:]

    self.tk  = tokenizer_bes.LangTokenizer()
    self.tk.load()

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    contx = self.tk.encode(row['eng'])
    input = [self.tk.sp.bos_id()] + self.tk.encode(row['ita'])
    label = (self.tk.encode(row['ita'])) + [self.tk.sp.eos_id()]
    return {
      'txt_eng': row['eng'],
      'txt_ita': row['ita'],
      'contx': torch.tensor(contx),
      'input': torch.tensor(input),
      'label': torch.tensor(label),
    }

  def collate_fn(self, batch):
    # Find the maximum length among all sequences in the batch
    max_contx_len = max(len(item['contx']) for item in batch)
    max_input_len = max(len(item['input']) for item in batch)
    max_label_len = max(len(item['label']) for item in batch)
    max_sequence = max(max_contx_len, max_input_len, max_label_len)

    # Manually pad each sequence to the maximum length
    contx_pad = torch.stack([torch.cat([item['contx'], torch.zeros(max_sequence - len(item['contx']))]) for item in batch], dim=0)
    input_pad = torch.stack([torch.cat([item['input'], torch.zeros(max_sequence - len(item['input']))]) for item in batch], dim=0)
    label_pad = torch.stack([torch.cat([item['label'], torch.zeros(max_sequence - len(item['label']))]) for item in batch], dim=0)

    return {
        'eng': [item['txt_eng'] for item in batch],
        'ita': [item['txt_ita'] for item in batch],
        'contx': contx_pad.to(torch.long),
        'input': input_pad.to(torch.long),
        'label': label_pad.to(torch.long),
    }

if __name__ == '__main__':
  # ds = Dataset()
  # emma = ds[0]
  # print('emma', emma)
  # 'plain': 'emma'
  # 'input': tensor([ 7, 15, 15,  3])
  # 'label': tensor([15, 15,  3,  1])
  # 'masks': tensor([ 1,  1,  1,  1])
  ds = LangDataset(data_split = 0.8)
  print('len(ds)', len(ds))
  print('ds[362]', ds[362])
