import argparse
from datetime import datetime

from models.transformer import Transformer

from util.utils import str2bool
from util.data import DataLoader
from util.tokenizer import get_tokenizer

import torch
import torch.nn as nn
from tqdm import tqdm

TIME = datetime.strftime(datetime.utcnow(), "%d-%m-%y_%H:%M")

class Trainer():
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # Training configuration
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, dest='lr')
        parser.add_argument('-de', '--weight_decay', type=float, default=1e-4, dest='weight_decay')
        parser.add_argument('-eps', type=float, default=3e-9, dest='optim_eps')
        parser.add_argument('-opt', '--optimizer', type=str, default='adam', dest='optimizer')
        parser.add_argument('-lw', '--load_weights', type=str, default=None, dest='load_name', help="Pretrained model weight file '.pth' or '.pt', default=None")
        parser.add_argument('-n_epochs', type=int, default=10, dest='num_epochs')
        parser.add_argument('-n_steps', type=int, default=50, dest='n_steps')
        parser.add_argument('-gpu', type=str2bool, default='true', dest='device')

        # Logging configuration
        parser.add_argument('-n', '--name', type=str, default=f"{TIME}.pth", dest="save_name")
        parser.add_argument('-c', "--checkpoints", type=str, default="checkpoints/", dest="save_dir")
        parser.add_argument('-lo', "--log_dir", type=str, default='logs/', dest="log_dir")
        
        # Data configuration
        parser.add_argument('-en_lang', type=str, default='.en', dest='en_lang')
        parser.add_argument('-de_lang', type=str, default='.de', dest='de_lang')
        parser.add_argument('-bs', '--batch_size', type=int, default=8, dest='batch_size')

        # Model configuration
        parser.add_argument('-d_model', type=int, default=512, dest='d_model')
        parser.add_argument('-d_ff', type=int, default=512, dest='d_ff')
        parser.add_argument('-max_len', type=int, default=256, dest='max_len')
        parser.add_argument('-drop_out', type=float, default=0.1, dest='drop_out')
        parser.add_argument('-n_layers', type=int, default=12, dest='n_layers')
        parser.add_argument('-n_heads', type=int, default=8, dest='n_heads')

        self.opt = parser.parse_args()
        if self.opt.device:
            self.opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.opt.device = 'cpu'
        self.init_loaders()
        self.init_model()
        self.init_optimizer()

        self.criterion = nn.CrossEntropyLoss()
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceOnPlateu()


    def init_loaders(self):
        dataloader = DataLoader(ext=(self.opt.en_lang, self.opt.de_lang),
                                tokenize_en=get_tokenizer(self.opt.en_lang),
                                tokenize_de=get_tokenizer(self.opt.de_lang),
                                init_token='<sos>',
                                eos_token='<eos>')
        
        train, valid, test = dataloader.make_dataset()
        dataloader.build_vocab(train_data=train, min_freq=2)
        self.train_iter, self.valid_iter, self.test_iter = dataloader.make_iter(train, valid, test,
                                                     batch_size=self.opt.batch_size,
                                                     device=self.opt.device)

        self.opt.src_pad_idx = dataloader.source.vocab.stoi['<pad>']
        self.opt.trg_pad_idx = dataloader.target.vocab.stoi['<pad>']
        self.opt.trg_sos_idx = dataloader.target.vocab.stoi['<sos>']
        print(self.opt.trg_pad_idx)
        print(self.opt.trg_sos_idx)
        print(dataloader.target.vocab.stoi['<eos>'])

        self.opt.enc_vocab_size = len(dataloader.source.vocab)
        self.opt.dec_vocab_size = len(dataloader.target.vocab)
    

    def init_model(self):
        self.model = Transformer(self.opt)
        if self.opt.load_name is not None:
            checkpoints = torch.load(self.opt.load_name)
            self.model.load_state_dict(checkpoints['weights'])
        
        self.model = self.model.to(self.opt.device)
    
    def init_optimizer(self):
        if self.opt.optimizer=='adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                              lr=self.opt.lr,
                                              weight_decay=self.opt.weight_decay,
                                              eps=self.opt.optim_eps)
    
    def init_logger(self):
        pass

    def train_epoch(self, epo):
        self.model.train()
        epo_loss = 0
        total = 0
        with tqdm(self.train_iter) as t:
            t.set_description(f'Training EPO {epo+1}:')
            for i, batch in enumerate(t):
                src, trg = batch.src, batch.trg
                # print("\n", trg[0], "\n", self.opt.trg_pad_idx)
                #NOTE: src and trg will be automatically padded via Transformer Embedding
                src, trg = src.to(self.opt.device), trg.to(self.opt.device)
                self.optimizer.zero_grad()
                output = self.model(src, trg) # Shift right, output: [B, seq_len, vocab_size]
                output = output.contiguous().view(-1, output.size(-1)) # Un-normalized Score Predictions
                trg = trg.contiguous().view(-1) # Ground truth class indices (classes refer to tokens)
                # label = torch.nn.functional.one_hot(trg, num_classes=self.opt.dec_vocab_size).float().to(self.opt.device)

                loss = self.criterion(output, trg)
                loss.backward()
                self.optimizer.step()

                epo_loss += loss.item()
                total += trg.size(0)
                t.set_postfix(loss=f"{epo_loss/total:.4f}")
        return epo_loss / total
    
    @torch.no_grad()
    def eval_epoch(self, epo):
        self.model.eval()
        epo_loss = 0
        total = 0
        with tqdm(self.valid_iter) as t:
            t.set_description(f"Eval EPO {epo+1}")
            for i, batch in enumerate(t):
                src, trg = batch.src, batch.trg
                src, trg = src.to(self.opt.device), trg.to(self.opt.device)
                output = self.model(src, trg[:, :-1])
                output = output.contiguous().view(-1, output.size(-1))
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, trg)

                epo_loss += loss.item()
                total += trg.size(0)
                t.set_postfix(loss=f"{epo_loss/total:.4f}")
        return epo_loss / total
    
    def train(self):
        for i in range(self.opt.num_epochs):
            trn_loss = self.train_epoch(i)

            val_loss = self.eval_epoch(i)

if __name__=="__main__":
    train_task = Trainer()
    train_task.train()
        