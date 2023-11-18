import sentencepiece as spm

class sp_tokenizer():
    def __init__(self, sp_model='ms_marco_model.model.model'):
        self.sp_model = sp_model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.sp_model)
    def encode(self,text):
        tokens = self.sp.encode_as_pieces(text)
        token_ids = [self.sp.piece_to_id(token) for token in tokens]
        return token_ids
    def decode(self,token_ids):
        decoded_text = self.sp.DecodeIds(token_ids)

        return decoded_text
        



if __name__ == '__main__':
    # Set the paths for your input text file and output model and vocab files
    input_file = 'corpus.txt'
    model_file = 'ms_marco_model.model'
    vocab_file = 'ms_marco_vocab.txt'

    # Train the SentencePiece model
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_file, vocab_size=16000)
    print('sentencepiece model and vocab files are saved')
