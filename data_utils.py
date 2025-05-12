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
    valid_target_chars = set("".join([l.strip().split(maxsplit=1)[1] for l in valid_lines]))

    
    source_chars = list(train_source_chars | valid_source_chars)
    target_chars = list(train_target_chars | valid_target_chars)
    
    extra_tokens = ["<s>", "</s>", "<unk>"]
    
    source_chars = extra_tokens + source_chars
    target_chars = extra_tokens + target_chars
    
    os.makedirs("dump", exist_ok=True)
    with open("dump/source_vocab.txt", "w", encoding="utf-8") as sv, open("dump/target_vocab.txt", "w", encoding="utf-8") as tv:
        sv.write("\n".join(source_chars))
        tv.write("\n".join(target_chars))
        
        
class Lang:
    def __init__(self, data_column, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as v:
            vocab_lines = v.read().splitlines()

        self.stoi_dict = {s: i+1 for i, s in enumerate(vocab_lines)}
        self.itos_dict = {v: k for k, v in self.stoi_dict.items()}

        self.SOS = self.stoi_dict.get("<s>")
        self.EOS = self.stoi_dict.get("</s>")
        self.UNK = self.stoi_dict.get("<unk>")

        self.data = data_column
        self.vocab_size = len(self.stoi_dict)+1

    def stoi(self, str_: str):
        return [self.stoi_dict.get(char, self.UNK) for char in str_]

    def itos(self, idxs: torch.Tensor):
        return "".join([self.itos_dict.get(idx.item(), "<unk>") for idx in idxs])

    def __getitem__(self, idx):
        word = self.data.iloc[idx]
        word_tokenized = self.stoi(word)
        return word_tokenized + [self.EOS]


class TranslitDataset:
    def __init__(self, data_path, normalize=False):
        assert data_path.endswith(".tsv")
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            
        if normalize:
            data = {}
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    source_word, target_word, n_attestation = parts
                    if not source_word in data.keys() or (source_word in data.keys() and n_attestation > data[source_word]["n_attestation"]):
                        data[source_word] = {"target":target_word, "n_attestation":n_attestation}
            data = [{"source":k, "target":v["target"]} for k, v in data.items()]
        else:
            data = []
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    data.append({"source":parts[0], "target":parts[1]})
                            
        self.df = pd.DataFrame(data)

        self.source = Lang(self.df["source"], "dump/source_vocab.txt")
        self.target = Lang(self.df["target"], "dump/target_vocab.txt")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s_idxs = self.source[idx]
        t_idxs = self.target[idx]
        return torch.tensor(s_idxs, dtype=torch.long), torch.tensor(t_idxs, dtype=torch.long)


def collate_fn(batch):
    source = [sample[0] for sample in batch]
    target = [sample[1] for sample in batch]

    source = pad_sequence(source, batch_first=True, padding_value=0)
    target = pad_sequence(target, batch_first=True, padding_value=0)

    return source, target

