import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from data_utils import collate_fn, TranslitDataset
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
from translit_model import TranslitModelConfig, TranslitModel



@dataclass
class TrainerConfig:
    trainset: TranslitDataset
    validset: TranslitDataset
    batch_size:int = 64,
    num_workers:int = 16,
    learning_rate:float = 0.003,
    weight_decay:float = 0.0005,
    teacher_forcing_p:float = 1.0,
    max_epoch:int = 10
    embedding_size: int = 256
    hidden_size: int = 256
    encoder_num_layers: int = 3
    decoder_num_layers: int = 2
    encoder_name: str = "GRU"
    decoder_name: str = "GRU"
    encoder_bidirectional: bool = True
    dropout_p: float = 0.3
    max_length: int = 32
    decoder_SOS: int = 0
    teacher_forcing_p: float = 0.8
    apply_attention: bool = True

    
class Trainer:
    def __init__(self, config:TrainerConfig):
        
        self.trainloader = DataLoader(config.trainset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=config.num_workers, persistent_workers=True, pin_memory=True)
        self.validloader = DataLoader(config.validset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=config.num_workers, persistent_workers=True, pin_memory=True)

        model_config = TranslitModelConfig(
            decoder_SOS=config.trainset.target.SOS,
            source_vocab_size=config.trainset.source.vocab_size,
            target_vocab_size=config.trainset.target.vocab_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            encoder_num_layers=config.encoder_num_layers,
            decoder_num_layers=config.decoder_num_layers,
            encoder_name=config.encoder_name,
            decoder_name=config.decoder_name,
            encoder_bidirectional=config.encoder_bidirectional,
            dropout_p=config.dropout_p,
            max_length=config.max_length,
            teacher_forcing_p=config.teacher_forcing_p,
            apply_attention=config.apply_attention,
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TranslitModel(model_config).to(self.device)
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


        print(self.model)
        numel = sum([param.numel() for param in self.model.parameters()])
        print("Number of parameters : ", numel)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.config = config


    def train(self):
        print("Starting Training ...")  
        for epoch in range(self.config.max_epoch):
            avg_loss, avg_acc = self.train_one_epoch(epoch)
            avg_valid_loss, avg_valid_acc = self.validate_one_epoch(epoch)
            print(f"Epoch : {epoch+1} | Train loss : {avg_loss:.4f} | Train accuracy : {avg_acc:.4f} | Valid loss : {avg_valid_loss:.4f} | Valid accuracy : {avg_valid_acc:.4f}")
        print("Training Finished !!!")
        

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_track = 0
        acc_track = 0
        for x, y in tqdm(self.trainloader, desc=f"Epoch {epoch+1}, Training "):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                out = self.model(x, y, teacher_forcing_p=self.config.teacher_forcing_p)
                loss, acc, _ = out
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_track += loss.item()
            acc_track += acc.item()
            
        avg_loss = loss_track/len(self.trainloader)
        avg_acc = acc_track/len(self.trainloader)
        return avg_loss, avg_acc
    
    def validate_one_epoch(self, epoch):
        self.model.eval()
        valid_loss_track, valid_acc_track = 0, 0
        with torch.no_grad():
            for x, y in tqdm(self.validloader, desc=f"Epoch {epoch+1}, Validation "):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                    out = self.model(x, y, teacher_forcing_p=0.0)
                    loss, acc, _ = out
                valid_loss_track += loss.item()
                valid_acc_track += acc.item()
            
            avg_valid_loss = valid_loss_track/len(self.validloader)
            avg_valid_acc = valid_acc_track/len(self.validloader)
        return avg_valid_loss, avg_valid_acc
    

    def inference(self, testset):
        print("Starting inference ...")
        testloader = DataLoader(testset, batch_size=self.config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=self.config.num_workers, persistent_workers=True, pin_memory=True)
        self.model.eval()
        with torch.no_grad():
            all_refs = []
            all_hyps = []
            for x, y in tqdm(testloader):
                x = x.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                    out = self.model(x, teacher_forcing_p=0)
                    _, _, decoder_outputs = out

                decoder_outputs = decoder_outputs.detach().cpu()
                y = y.cpu()

                probs = F.log_softmax(decoder_outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_refs.extend(list(y))
                all_hyps.extend(list(preds))

            bleu_, targets, preds = self.scoring(all_refs, all_hyps)
            
            idx = list(range(len(targets)))
            random.shuffle(idx)
            
            print("BLEU score:", bleu_)
            
            print("Samples:\n")
            print("Targets\tPredictions")
            for i in idx[:20]:
                print(f"{targets[i]}\t{preds[i]}") 
 

    def scoring(self, refs, hyps):
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
