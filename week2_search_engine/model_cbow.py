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