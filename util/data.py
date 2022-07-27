from torchtext.data import Field, BucketIterator
from torchtext.datasets.translation import Multi30k

from util.tokenizer import get_tokenizer

class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
        self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True) # Included padding

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator


if __name__=="__main__":
    loader= DataLoader(('.en', '.de'), get_tokenizer('en'), get_tokenizer('fr'),
                                    '<sos>', '<eos>')

    train, val, test = loader.make_dataset()
    loader.build_vocab(train, 2)
    train_iter, val_iter, test_iter = loader.make_iter(train, val, test, 8, 'cpu')

    for i, batch in enumerate(train_iter):
        src = batch.src
        print(src.size())
        if i == 3:
            break