import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

    

def preprocessor(train_data_path, valid_data_path):
    with open(train_data_path, "r", encoding="utf-8") as ft, open(valid_data_path, "r", encoding="utf-8") as fv:
        train_lines = ft.read().splitlines()
        valid_lines = fv.read().splitlines()
            
    train_source_chars = set("".join([l.strip().split(maxsplit=1)[0] for l in train_lines]))
    train_target_chars = set("".join([l.strip().split(maxsplit=1)[1] for l in train_lines]))
    valid_source_chars = set("".join([l.strip().split(maxsplit=1)[0] for l in valid_lines]))
    valid_target_chars = set("".join([l.strip().split(maxsplit=1)[1] for l in train_lines]))
    
    source_chars = list(train_source_chars | valid_source_chars)
    target_chars = list(train_target_chars | valid_target_chars)
    
    extra_tokens = ["<s>", "</s>", "<unk>"]
    
    source_chars = extra_tokens + source_chars
    target_chars = extra_tokens + target_chars
    
    os.makedirs("dump", exist_ok=True)
    with open("dump/source_vocab.txt", "w", encoding="utf-8") as sv, open("dump/target_vocab.txt", "w", encoding="utf-8") as tv:
        sv.write("\n".join(source_chars))
        tv.write("\n".join(target_chars))
        

class TranslitDataset:
    def __init__(self, data_path):
        assert data_path.endswith(".tsv")
        self.df = pd.read_csv(data_path, sep="\t", header=None, usecols=[0, 1], names=["source", "target"])
        
        with open("dump/source_vocab.txt", "r", encoding="utf-8") as sv, open("dump/target_vocab.txt", "r", encoding="utf-8") as tv:
            sv_lines = sv.read().splitlines()
            tv_lines = tv.read().splitlines()
        
        self.source_stoi = {s:i+1 for i, s in enumerate(sv_lines)}
        self.target_stoi = {s:i+1 for i, s in enumerate(tv_lines)}
        self.source_itos = {v:k for k, v in self.source_stoi.items()}
        self.target_itos = {v:k for k, v in self.target_stoi.items()}

        
    def stoi(self, str_: str, stoi_map: dict):
        return [stoi_map.get(char, "<unk>") for char in str_]
    
    def itos(self, idxs: torch.Tensor, itos_map: dict):
        return "".join([itos_map[idx.item()] for idx in idxs])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s_idxs = self.stoi(self.df.iloc[idx]["source"], stoi_map=self.source_stoi) + [self.source_stoi["</s>"]]
        t_idxs = self.stoi(self.df.iloc[idx]["target"], stoi_map=self.target_stoi) + [self.target_stoi["</s>"]]
        return torch.tensor(s_idxs, dtype=torch.long), torch.tensor(t_idxs, dtype=torch.long)
    
    

def collate_fn(batch):
    source = [sample[0] for sample in batch]
    target = [sample[1] for sample in batch]
    
    source = pad_sequence(source, batch_first=True, padding_value=0)
    target = pad_sequence(target, batch_first=True, padding_value=0)
    
    return source, target

    


# for x, y in trainloader:
#     print(x.shape, y.shape)
#     print(x)
#     print(y)
#     break