import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from data_utils import preprocessor, TranslitDataset, collate_fn
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random

@dataclass
class TranslitModelConfig:
    source_vocab_size: int = 500
    target_vocab_size: int = 500
    embedding_size: int = 256
    hidden_size: int = 512
    encoder_num_layers: int = 4
    decoder_num_layers: int = 3
    encoder_name: str = "GRU"
    decoder_name: str = "GRU"
    encoder_bidirectional: bool = True
    decoder_bidirectional: bool = True
    dropout_p: float = 0.3
    max_length: int = 64
    decoder_SOS: int = 0
    teacher_forcing_p: float = 1 # only works for 1 for now

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


class EncoderAttnRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
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
        N, max_length = x.shape
        D = 2 if self.args.encoder_bidirectional else 1
        hidden_state = torch.zeros(D * self.args.encoder_num_layers, N, self.args.hidden_size, device=x.device)

        encoder_hidden_states = []
        encoder_outs = []
        for i in range(max_length):
            out, hidden_state = self.forward_one_step(x[:, i], hidden_state)
            encoder_outs.append(out.squeeze(1))
            encoder_hidden_states.append(hidden_state)

        encoder_outs = torch.stack(encoder_outs, dim=1) # N, L, Hout
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=2) # (D * num_layers, N, L, Hout)
        return encoder_outs, encoder_hidden_states # returned as a list for now

    def forward_one_step(self, input_, hidden_state):
        input_ = input_.unsqueeze(1)
        embeds = self.dropout(self.embedding(input_))
        output, hidden = self.rnn(embeds, hidden_state)
        output = self.final_dropout(output)
        if isinstance(hidden, tuple): 
            h_n, _ = hidden
        else:
            h_n = hidden
        return output, h_n
        

class DecoderAttnRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pass

    def forward(self, encoder_outputs, encoder_hidden_states, target=None):
        
        
        
    
    def forward_one_step(self, decoder_input, decoder_hidden):
        pass
    



class Projection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_D = 2 if args.encoder_bidirectional else 1
        self.decoder_D = 2 if args.decoder_bidirectional else 1
        encoder_out_shape = self.encoder_D * args.encoder_num_layers * args.hidden_size
        decoder_in_shape = self.decoder_D * args.decoder_num_layers * args.hidden_size
        self.linear = nn.Linear(encoder_out_shape, decoder_in_shape)
    
    def forward(self, x):
        B = x.shape[1]
        x = x.transpose(0, 1).contiguous().view(B, -1)
        x_proj = self.linear(x)
        x_proj = x_proj.view(self.decoder_D * self.args.decoder_num_layers, B, self.args.hidden_size)
        return x_proj
        

class TranslitModel(nn.Module):
    def __init__(self, args: TranslitModelConfig):
        super().__init__()
        self.args = args
        self.encoder = EncoderRNN(args)
        self.decoder = DecoderRNN(args)
        self.loss_fn = nn.CrossEntropyLoss()
    
    
    def forward(self, source, target=None):
        encoder_outputs, encoder_hidden_final = self.encoder(source)
        decoder_expected_size = 2*self.args.decoder_num_layers if self.args.decoder_bidirectional else self.args.decoder_num_layers
        if decoder_expected_size <= encoder_hidden_final.shape[0]:
            encoder_hidden_final = encoder_hidden_final[:decoder_expected_size, :, :]
        else:
            diff = decoder_expected_size - encoder_hidden_final.shape[0]
            encoder_hidden_final = torch.cat((encoder_hidden_final, encoder_hidden_final[:diff]), dim=0)
            
        decoder_outputs = self.decoder(encoder_outputs, encoder_hidden_final, target)
        
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


trainloader = DataLoader(trainset, batch_size=128, collate_fn=collate_fn, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
validloader = DataLoader(validset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
testloader = DataLoader(testset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

config = TranslitModelConfig(
    decoder_SOS=trainset.target.SOS,
    source_vocab_size=trainset.source.vocab_size,
    target_vocab_size=trainset.target.vocab_size
    )


device = "cuda" if torch.cuda.is_available() else "cpu"
model = TranslitModel(config).to(device)
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


print(model)
numel = sum([param.numel() for param in model.parameters()])
print("Number of parameters : ", numel)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
max_epochs=5

for epoch in range(max_epochs):
    model.train()
    loss_track = 0
    acc_track = 0
    for x, y in tqdm(trainloader, desc=f"Epoch {epoch+1}, Training "):
        x = x.to(device, non_blocking=True)
        
        if torch.rand(1) <= config.teacher_forcing_p:
            y = y.to(device, non_blocking=True)
        else:
            y = None
        
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            out = model(x, y)
            loss, acc, _ = out
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_track += loss.item()
        acc_track += acc.item()
        
    avg_loss = loss_track/len(trainloader)
    avg_acc = acc_track/len(trainloader)
    
    model.eval()
    valid_loss_track, valid_acc_track = 0, 0
    with torch.no_grad():
        for x, y in tqdm(validloader, desc=f"Epoch {epoch+1}, Validation "):
            x = x.to(device, non_blocking=True)
            y = None
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                out = model(x, y)
                loss, acc, _ = out
            valid_loss_track += loss.item()
            valid_acc_track += acc.item()
        
        avg_valid_loss = valid_loss_track/len(validloader)
        avg_valid_acc = valid_acc_track/len(validloader)

    print(f"Epoch : {epoch+1} | Train loss : {avg_loss:.4f} | Train accuracy : {avg_acc:.4f} | Valid loss : {avg_valid_loss:.4f} | Valid accuracy : {avg_valid_acc:.4f}")
        
      

def BleuScore(refs, hyps):
    total_bleu = 0
    count = 0
    targets = []
    preds = []
    for ref_seq, hyp_seq in zip(refs, hyps):
        eos_idx_ref = (ref_seq == testset.target.EOS).nonzero(as_tuple=True)[0]
        eos_idx_hyp = (hyp_seq == testset.target.EOS).nonzero(as_tuple=True)[0]
        ref_trimmed = ref_seq[:eos_idx_ref[0].item()] if len(eos_idx_ref) > 0 else ref_seq
        hyp_trimmed = hyp_seq[:eos_idx_hyp[0].item()] if len(eos_idx_hyp) > 0 else hyp_seq

        ref_text = testset.target.itos(ref_trimmed)
        hyp_text = testset.target.itos(hyp_trimmed)
        
        targets.append(ref_text)
        preds.append(hyp_text)
        
        ref_text_split = list(ref_text)
        hyp_text_split = list(hyp_text)
        
        bleu = sentence_bleu([ref_text_split], hyp_text_split, smoothing_function=SmoothingFunction().method1)
        total_bleu += bleu
        count += 1

    bleu = total_bleu / count if count > 0 else 0.0
    return bleu, targets, preds

        
        
# Inference:
model.eval()
with torch.no_grad():
    all_refs = []
    all_hyps = []
    for x, y in tqdm(testloader, desc=f"Epoch {epoch+1}, Inference"):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            out = model(x)
            _, _, decoder_outputs = out

        decoder_outputs = decoder_outputs.detach().cpu()
        y = y.cpu()

        probs = F.log_softmax(decoder_outputs, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_refs.extend(list(y))
        all_hyps.extend(list(preds))

    score, targets, preds = BleuScore(all_refs, all_hyps)
    
    idx = list(range(len(targets)))
    random.shuffle(idx)
    
    print("BLEU score:", score)
    
    print("Samples:\n")
    print("Targets\tPredictions")
    for i in idx[:20]:
        print(f"{targets[i]}\t{preds[i]}")    
    
