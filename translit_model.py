import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from data_utils import preprocessor, TranslitDataset, collate_fn
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
from collections import namedtuple

@dataclass
class TranslitModelConfig:
    source_vocab_size: int = 500
    target_vocab_size: int = 500
    embedding_size: int = 256
    hidden_size: int = 256
    encoder_num_layers: int = 3
    decoder_num_layers: int = 2
    encoder_name: str = "GRU"
    decoder_name: str = "GRU"
    encoder_bidirectional: bool = True
    decoder_bidirectional: bool = False
    dropout_p: float = 0.3
    max_length: int = 32
    decoder_SOS: int = 0
    teacher_forcing_p: float = 0.8
    apply_attention: bool = True

    def __post_init__(self):
        assert self.encoder_name in ["RNN", "GRU", "LSTM"], f"Invalid encoder name: {self.encoder_name}. Must be 'RNN', 'LSTM', or 'GRU'."
        assert self.decoder_name in ["RNN", "GRU", "LSTM"], f"Invalid decoder name: {self.decoder_name}. Must be 'RNN', 'LSTM', or 'GRU'."


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
            bidirectional=False
        )
        classifier_in_dim = args.hidden_size
        self.classifier_head = nn.Linear(classifier_in_dim, args.target_vocab_size)

    def forward(self, encoder_outputs, encoder_hidden, target=None, teacher_forcing_p=1.0, beam_size=1, return_attention_map=False):
        decoder_hidden = encoder_hidden
        max_length = target.shape[1] if target is not None else self.args.max_length
        
        if torch.rand(1) > teacher_forcing_p:
            target = None
            
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
        return decoder_outputs, None


    def forward_one_step(self, decoder_input, decoder_hidden):
        assert decoder_input.shape[1] == 1
        embeds = self.relu(self.embedding(decoder_input))
        embeds = self.dropout(embeds)
        if self.args.decoder_name == "LSTM":
            decoder_hidden = (decoder_hidden, None)
        output, hidden = self.rnn(embeds, decoder_hidden)
        if isinstance(hidden, tuple):
            h_n, _ = hidden
        else:
            h_n = hidden
        output = self.classifier_head(output)
        return output, h_n


class BahdanauAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.We = nn.Linear(args.hidden_size*2, args.hidden_size) if args.encoder_bidirectional else nn.Linear(args.hidden_size, args.hidden_size)
        self.Wd = nn.Linear(args.hidden_size, args.hidden_size)
        self.Wo = nn.Linear(args.hidden_size, 1)
        self.non_linearity = nn.Tanh()
        
    def forward(self, encoder_outputs, decoder_hidden):
        
        # print(decoder_hidden.shape, encoder_outputs.shape)
        B, T_enc, H_enc = encoder_outputs.size()
        decoder_hidden = decoder_hidden[-1]
        
        encoder_features = self.We(encoder_outputs)
        decoder_features = self.Wd(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(encoder_features + decoder_features)
        scores = self.Wo(energy).squeeze(-1)

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights


class DecoderAttnRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.target_vocab_size, args.embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_p)
        self.attention = BahdanauAttention(args)
        D = 2 if args.encoder_bidirectional else 1
        rnn_input = args.embedding_size + (D * args.hidden_size)
        self.rnn = getattr(nn, args.decoder_name)(
            input_size=rnn_input,
            hidden_size=args.hidden_size,
            num_layers=args.decoder_num_layers,
            batch_first=True,
            dropout=args.dropout_p if args.decoder_num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.classifier_head = nn.Linear(args.hidden_size, args.target_vocab_size)
        self.BeamHypothesis = namedtuple("BeamHypothesis", ["tokens", "logprob", "hidden"])


    def forward(self, encoder_outputs, encoder_hidden, target=None, teacher_forcing_p=1.0, beam_size=1, return_attention_map=False):
        if beam_size == 1:
            return self.greedy_decode(encoder_outputs, encoder_hidden, target, teacher_forcing_p, return_attention_map)
        else:
            return self.beam_search_decode(encoder_outputs, encoder_hidden, beam_size)

    def greedy_decode(self, encoder_outputs, encoder_hidden, target, teacher_forcing_p, return_attention_map):
        decoder_hidden = encoder_hidden
        max_length = target.shape[1] if target is not None else self.args.max_length
        if torch.rand(1) > teacher_forcing_p:
            target = None

        decoder_input = torch.full((encoder_outputs.size(0), 1), fill_value=self.args.decoder_SOS,
                                   dtype=torch.long, device=encoder_outputs.device)
        decoder_outputs = []
        attention_map = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attention_map_one_step = self.forward_one_step(decoder_input, decoder_hidden, encoder_outputs, return_attention_map)
            
            # if return_attention_map:
            #     print(attention_map_one_step)
            
            if attention_map_one_step is not None:
                attention_map.append(attention_map_one_step)
            
            decoder_outputs.append(decoder_output)

            if target is not None:
                decoder_input = target[:, i].unsqueeze(1)
            else:
                probs = F.log_softmax(decoder_output, dim=-1)
                decoder_input = torch.argmax(probs, dim=-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attention_map = torch.stack(attention_map)

        # print("attention_map shape : ", attention_map.shape)
        return decoder_outputs, attention_map

    def forward_one_step(self, decoder_input, decoder_hidden, encoder_outputs, return_attention_map=False):
        embeds = self.relu(self.embedding(decoder_input))
        embeds = self.dropout(embeds)
        context, probs = self.attention(encoder_outputs, decoder_hidden)
        input_ = torch.cat((embeds, context), dim=2)

        if self.args.decoder_name == "LSTM" and not isinstance(decoder_hidden, tuple):
            decoder_hidden = (decoder_hidden, torch.zeros_like(decoder_hidden))

        output, hidden = self.rnn(input_, decoder_hidden)
        logits = self.classifier_head(output)
        return logits, hidden, probs

    def beam_search_decode(self, encoder_outputs, encoder_hidden, beam_size):
        batch_size = encoder_outputs.size(0)
        max_len = self.args.max_length
        device = encoder_outputs.device
        vocab_size = self.args.target_vocab_size

        decoder_input = torch.full((batch_size, 1), self.args.decoder_SOS, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        
        sequences = []
        for b in range(batch_size):
            hidden_b = decoder_hidden[:, b:b+1, :].contiguous()
            sequences.append([self.BeamHypothesis(tokens=[self.args.decoder_SOS], logprob=0.0, hidden=hidden_b) for _ in range(beam_size)])
        
        for t in range(max_len):
            all_candidates = [[] for _ in range(batch_size)]
            for b in range(batch_size):
                for hyp in sequences[b]:
                    last_token = torch.tensor([[hyp.tokens[-1]]], device=device)
                    hidden = hyp.hidden
                    logits, next_hidden, _ = self.forward_one_step(last_token, hidden, encoder_outputs[b:b+1])
                    log_probs = F.log_softmax(logits.squeeze(1), dim=-1)

                    top_log_probs, top_idxs = torch.topk(log_probs, beam_size)
                    for log_p, idx in zip(top_log_probs[0], top_idxs[0]):
                        new_tokens = hyp.tokens + [idx.item()]
                        new_logprob = hyp.logprob + log_p.item()
                        all_candidates[b].append(self.BeamHypothesis(tokens=new_tokens, logprob=new_logprob, hidden=next_hidden))

                ordered = sorted(all_candidates[b], key=lambda h: h.logprob, reverse=True)
                sequences[b] = ordered[:beam_size]

        output_sequences = []
        for b in range(batch_size):
            best_seq = max(sequences[b], key=lambda h: h.logprob)
            output_sequences.append(torch.tensor(best_seq.tokens, device=device).unsqueeze(0))

        padded_outputs = torch.full((batch_size, max_len), fill_value=0, dtype=torch.long, device=device)
        for i, seq in enumerate(output_sequences):
            seq = seq[:, :max_len]
            padded_outputs[i, :seq.size(1)] = seq

        return padded_outputs, None


class TranslitModel(nn.Module):
    def __init__(self, args: TranslitModelConfig):
        super().__init__()
        self.args = args
        self.encoder = EncoderRNN(args)
        self.decoder = DecoderAttnRNN(args) if args.apply_attention else DecoderRNN(args)
        self.loss_fn = nn.CrossEntropyLoss()
    
    
    def forward(self, source, target=None, teacher_forcing_p=1.0, beam_size=1, return_attention_map=False):
        encoder_outputs, encoder_hidden_final = self.encoder(source)
        decoder_expected_size = 2*self.args.decoder_num_layers if self.args.decoder_bidirectional else self.args.decoder_num_layers
        if decoder_expected_size <= encoder_hidden_final.shape[0]:
            encoder_hidden_final = encoder_hidden_final[:decoder_expected_size, :, :]
        else:
            diff = decoder_expected_size - encoder_hidden_final.shape[0]
            encoder_hidden_final = torch.cat((encoder_hidden_final, encoder_hidden_final[:diff]), dim=0)
        
        if return_attention_map:
            assert source.shape[0] == 1, f"attention map is returned only for batch size 1, but found {source.shape[0]}"
        decoder_outputs, attention_map = self.decoder(encoder_outputs, encoder_hidden_final, target, teacher_forcing_p=teacher_forcing_p, beam_size=beam_size, return_attention_map=return_attention_map)
        
        loss, acc = None, None
        if target is not None:
            loss, acc = self.compute_loss_and_acc(decoder_outputs, target)
        return loss, acc, decoder_outputs, attention_map
    
    
    def compute_loss_and_acc(self, decoder_outputs, target):        
        target_flattend = target.flatten()
        _, _, n_classes = decoder_outputs.shape
        deocder_outputs_flattened = decoder_outputs.view(-1, n_classes)
        loss = self.loss_fn(deocder_outputs_flattened, target_flattend)
        probs = F.log_softmax(deocder_outputs_flattened, dim=-1)
        max_ = torch.argmax(probs, dim=-1)
        acc = (max_ == target_flattend).sum() / len(target_flattend)
        return loss, acc
    

        
        