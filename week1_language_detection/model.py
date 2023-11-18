import torch


class CBOW(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CBOW, self).__init__()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.linear = torch.nn.Linear(embedding_dim, vocab_size)

  def forward(self, inputs):
    embeds = torch.sum(self.embeddings(inputs), dim=1)
    out = self.linear(embeds)
    log_probs = torch.nn.functional.log_softmax(out, dim=1)
    return log_probs


class SkipGram(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(SkipGram, self).__init__()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.linear = torch.nn.Linear(embedding_dim, vocab_size)

  def forward(self, target_word):
    embeds = self.embeddings(target_word)
    out = self.linear(embeds)
    log_probs = torch.nn.functional.log_softmax(out, dim=1)
    return log_probs


class Language(torch.nn.Module):
  def __init__(self, embedding_weights, num_classes=7):
    super(Language, self).__init__()
    vocab_size, embedding_dim = embedding_weights.size()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.embeddings.load_state_dict({'weight': embedding_weights})
    # self.embeddings.requires_grad = False  # Optional: Freeze embeddings
    self.linear = torch.nn.Linear(embedding_dim, num_classes)
    self.relu = torch.nn.ReLU()

  def forward(self, inputs):
    embeds = self.embeddings(inputs)
    # Average pooling along the sequence length
    pooled = torch.mean(embeds, dim=1)
    output = self.linear(pooled)
    output = self.relu(output)
    return output


if __name__ == '__main__':
  cbow = CBOW(70000, 50)
  emb_weights = cbow.embeddings.weight.data # shape(20.000, 50)
  lang = Language(emb_weights)
  # Open and read the text file
  word_dict = {}

  # Open and read the text file with UTF-8 encoding
  with open("vocab.txt", "r", encoding="utf-8") as file:
      for line_number, line in enumerate(file, start=1):
          # Remove newline characters and leading/trailing whitespaces
          word = line.strip()
          
          # Add the word to the dictionary with line number as the key
          word_dict[word] = line_number

  tokens = ['je','chat','voiture']
  sentence = []
  for token in tokens:
    if token in word_dict.keys():
      sentence.append(word_dict[token])
  sentence = torch.tensor([sentence]) # shape(1, 3)
  
  out = lang(sentence) # shape(1, 7)
