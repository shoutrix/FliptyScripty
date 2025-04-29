import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from data_utils import preprocessor, TranslitDataset, collate_fn

@dataclass
class TranslitModelConfig:
    source_vocab_size: int = 500
    target_vocab_size: int = 500
    embedding_size: int = 256
    hidden_size: int = 512
    encoder_num_layers: int = 2
    decoder_num_layers: int = 2
    encoder_name: str = "GRU"   # options: "rnn", "lstm", "gru"
    decoder_name: str = "GRU"   # options: "rnn", "lstm", "gru"
    encoder_bidirectional: bool = True
    decoder_bidirectional: bool = True
    dropout_p: float = 0.3
    max_length: int = 64

    def __post_init__(self):
        assert self.encoder_name in ["rnn", "lstm", "GRU"], f"Invalid encoder name: {self.encoder_name}. Must be 'rnn', 'lstm', or 'gru'."
        assert self.decoder_name in ["rnn", "lstm", "GRU"], f"Invalid decoder name: {self.decoder_name}. Must be 'rnn', 'lstm', or 'gru'."


class EncoderRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.source_vocab_size, args.embedding_size)
        self.dropout = nn.Dropout(args.dropout_p)
        self.rnn = getattr(nn, args.encoder_name)(
            input_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.encoder_num_layers,
            batch_first=True,
            dropout=args.dropout_p if args.encoder_num_layers > 1 else 0.0,
            bidirectional=args.encoder_bidirectional
        )
        self.final_dropout = nn.Dropout(args.dropout_p)

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embeds)
        output = self.final_dropout(output)
        
        if isinstance(hidden, tuple): 
            h_n, _ = hidden
        else:
            h_n = hidden
        return output, h_n


class DecoderRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.target_vocab_size, args.embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_p)
        self.rnn = getattr(nn, args.decoder_name)(
            input_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.decoder_num_layers,
            batch_first=True,
            dropout=args.dropout_p if args.decoder_num_layers > 1 else 0.0,
            bidirectional=args.decoder_bidirectional
        )
        self.classifier_head = nn.Linear(args.hidden_size, args.target_vocab_size)

    def forward(self, encoder_outputs, encoder_hidden, target=None):
        
        
        
        
        decoder_outputs = []
        decoder_output, decoder_hidden = self.forward_one_step(decoder_input, decoder_hidden)
        
    
    def forward_one_step(self, decoder_input, decoder_hidden):
        pass
        
        
        


class TranslitModel(nn.Module):
    def __init__(self, args: TranslitModelConfig):
        super().__init__()
        self.args = args
        self.encoder = EncoderRNN(args)
        self.decoder = DecoderRNN(args)
    
    def forward(self, source, target=None):
        encoder_outputs, encoder_hidden = self.encoder(source)
        print("encoder_outputs : ", encoder_outputs.shape, encoder_hidden.shape)
        loss, acc, decoder_outputs = self.decoder(encoder_outputs, encoder_hidden, target)
        return loss, acc, decoder_outputs


config = TranslitModelConfig()
model = TranslitModel(config)

train_data_path = "/speech/shoutrik/torch_exp/FliptyScripty/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"
dev_data_path = "/speech/shoutrik/torch_exp/FliptyScripty/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
test_data_path = "/speech/shoutrik/torch_exp/FliptyScripty/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv"

preprocessor(train_data_path, dev_data_path)
trainset = TranslitDataset(train_data_path)
validset = TranslitDataset(dev_data_path)
testset = TranslitDataset(test_data_path)

trainloader = DataLoader(trainset, batch_size=4, collate_fn=collate_fn, shuffle=True)


for x, y in trainloader:
    print(x.shape, y.shape)
    # print(x)
    # print(y)
    out = model(x)
    loss, acc, decoder_outputs = out
    print(loss)
    print(acc)
    print(decoder_outputs.shape)
    print("\n\n")


