import sp_tokenizer
import torch

class W2VData(torch.utils.data.Dataset):
    def __init__(self, corpus,window_size=2):
        self.data = []
        self.generate_CBOW_ds(corpus,window_size)

    def generate_CBOW_ds(self,corpus, window_size):
        tokenizer = sp_tokenizer.sp_tokenizer()
        for line in corpus:
            tokens = tokenizer.encode(line)
            for i, target in enumerate(tokens):
                context = tokens[max(0, i - window_size):i] + tokens[i + 1:i + window_size + 1]
                if len(context) != 2 * window_size: continue
                self.data.append((torch.tensor(context), torch.tensor(target)))