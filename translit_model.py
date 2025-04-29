import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from data_utils import preprocessor, TranslitDataset, collate_fn
import torch.nn.functional as F
from tqdm import tqdm

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
    decoder_SOS: int = 0

    def __post_init__(self):
        assert self.encoder_name in ["RNN", "GRU"], f"Invalid encoder name: {self.encoder_name}. Must be 'rnn', 'lstm', or 'gru'."
        assert self.decoder_name in ["RNN", "GRU"], f"Invalid decoder name: {self.decoder_name}. Must be 'rnn', 'lstm', or 'gru'."


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
        classifier_in_dim = args.hidden_size * (2 if args.decoder_bidirectional else 1)
        self.classifier_head = nn.Linear(classifier_in_dim, args.target_vocab_size)

    def forward(self, encoder_outputs, encoder_hidden, target=None):
        decoder_hidden = encoder_hidden
        max_length = target.shape[1] if target is not None else self.args.max_length

        decoder_input = torch.full((encoder_outputs.shape[0], 1), fill_value=self.args.decoder_SOS, dtype=torch.long, device=encoder_outputs.device)

        decoder_outputs = []
        for i in range(max_length):
            decoder_output, decoder_hidden = self.forward_one_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target is not None:
                decoder_input = target[:, i].unsqueeze(1)
            else:
                probs = F.log_softmax(decoder_output, dim=-1)
                decoder_input = torch.argmax(probs, dim=-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs


    def forward_one_step(self, decoder_input, decoder_hidden):
        assert decoder_input.shape[1] == 1
        embeds = self.relu(self.embedding(decoder_input))
        embeds = self.dropout(embeds)
        output, hidden = self.rnn(embeds, decoder_hidden)
        if isinstance(hidden, tuple):
            h_n, _ = hidden
        else:
            h_n = hidden
        output = self.classifier_head(output)
        return output, h_n




class TranslitModel(nn.Module):
    def __init__(self, args: TranslitModelConfig):
        super().__init__()
        self.args = args
        self.encoder = EncoderRNN(args)
        self.decoder = DecoderRNN(args)
        self.loss_fn = nn.CrossEntropyLoss()
    
    
    def forward(self, source, target=None):
        encoder_outputs, encoder_hidden = self.encoder(source)
        # print("encoder_outputs : ", encoder_outputs.shape, encoder_hidden.shape)
        decoder_outputs = self.decoder(encoder_outputs, encoder_hidden, target)
        
        loss, acc = None, None
        if target is not None:
            loss, acc = self.compute_loss_and_acc(decoder_outputs, target)
        return loss, acc, decoder_outputs
    
    
    def compute_loss_and_acc(self, decoder_outputs, target):
        target_flattend = target.flatten()
        _, _, n_classes = decoder_outputs.shape
        deocder_outputs_flattened = decoder_outputs.view(-1, n_classes)
        loss = self.loss_fn(deocder_outputs_flattened, target_flattend)
        probs = F.log_softmax(deocder_outputs_flattened, dim=-1)
        max_ = torch.argmax(probs, dim=-1)
        acc = (max_ == target_flattend).sum() / len(target_flattend)
        return loss, acc



train_data_path = "/speech/shoutrik/torch_exp/FliptyScripty/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"
dev_data_path = "/speech/shoutrik/torch_exp/FliptyScripty/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv"
test_data_path = "/speech/shoutrik/torch_exp/FliptyScripty/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv"

preprocessor(train_data_path, dev_data_path)
trainset = TranslitDataset(train_data_path)
validset = TranslitDataset(dev_data_path)
testset = TranslitDataset(test_data_path)


trainloader = DataLoader(trainset, batch_size=128, collate_fn=collate_fn, shuffle=True)

config = TranslitModelConfig(
    decoder_SOS=trainset.target.SOS,
    source_vocab_size=trainset.source.vocab_size,
    target_vocab_size=trainset.target.vocab_size
    )

device = "cuda"

model = TranslitModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
max_epochs=10

for epoch in range(max_epochs):
    loss_track = 0
    acc_track = 0
    for x, y in tqdm(trainloader):
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)
        loss, acc, decoder_outputs = out
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_track += loss.item()
        acc_track += acc.item()
        
    avg_loss = loss_track/len(trainloader)
    avg_acc = acc_track/len(trainloader)

    print(f"Epoch : {epoch+1} | Loss : {avg_loss:.4f} | Accuracy : {avg_acc:.4f}")
        
        
        
        



