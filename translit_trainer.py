import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from data_utils import collate_fn, TranslitDataset, preprocessor
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
from translit_model import TranslitModelConfig, TranslitModel
import os
import wandb
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from q6 import make_q6


nirm = '/speech/shoutrik/torch_exp/FliptyScripty/Assignment3/Nirmala.ttf'


lang_map = {
    "bengali": "bn",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "punjabi": "pa",
    "sindhi": "sd",
    "sinhala": "si",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur"}

@dataclass
class TrainerConfig:
    language:str = "hindi",
    data_path:str = "",
    batch_size:int = 256,
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
    beam_size: int = 1
    logging: bool = False
    
    
    def __post_init__(self):
        if self.language not in lang_map:
            raise ValueError(f"Unsupported language '{self.language}'. Supported options: {list(lang_map.keys())}")
        self.lang_code = lang_map[self.language]
        self.train_data_path = os.path.join(self.data_path, f"{self.lang_code}/lexicons/{self.lang_code}.translit.sampled.train.tsv")
        self.dev_data_path = os.path.join(self.data_path, f"{self.lang_code}/lexicons/{self.lang_code}.translit.sampled.dev.tsv")
        self.test_data_path = os.path.join(self.data_path, f"{self.lang_code}/lexicons/{self.lang_code}.translit.sampled.test.tsv")
        if not all(os.path.exists(path) for path in [self.train_data_path, self.dev_data_path, self.test_data_path]):
            raise FileNotFoundError(f"Data files not found in {self.data_path}. Please check the paths.")
            
        
class Trainer:
    def __init__(self, config:TrainerConfig, logging):

        preprocessor(config.train_data_path, config.dev_data_path)

        trainset = TranslitDataset(config.train_data_path, normalize=True)
        validset = TranslitDataset(config.dev_data_path, normalize=False)   
        testset = TranslitDataset(config.test_data_path, normalize=False)
        
        # print("NUM_WORKERE : ", config.num_workers)
        
        self.trainloader = DataLoader(trainset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=config.num_workers, persistent_workers=True, pin_memory=True)
        self.validloader = DataLoader(validset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=config.num_workers, persistent_workers=True, pin_memory=True)
        self.testloader = DataLoader(testset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=config.num_workers, persistent_workers=True, pin_memory=True)

        model_config = TranslitModelConfig(
            decoder_SOS=trainset.target.SOS,
            source_vocab_size=trainset.source.vocab_size,
            target_vocab_size=trainset.target.vocab_size,
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
        print(f"Setting device to : {self.device}")
        self.model = TranslitModel(model_config).to(self.device)
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("Setting autocast dtype to : ", self.autocast_dtype)
        print("Model architecture : ")
        print(self.model)
        numel = sum([param.numel() for param in self.model.parameters()])
        print("Number of parameters : ", numel)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.config = config
        self.logging = logging


    def train(self):
        print("Starting Training ...")  
        for epoch in range(self.config.max_epoch):
            avg_loss, avg_acc = self.train_one_epoch(epoch)
            avg_valid_loss, avg_valid_acc = self.validate_one_epoch(epoch)
            print(f"Epoch : {epoch+1} | Train loss : {avg_loss:.4f} | Train accuracy : {avg_acc:.4f} | Valid loss : {avg_valid_loss:.4f} | Valid accuracy : {avg_valid_acc:.4f}")
            if self.logging:
                wandb.log({
                    "train_loss": avg_loss,
                    "train_accuracy": avg_acc,
                    "valid_loss": avg_valid_loss,
                    "valid_accuracy": avg_valid_acc,
                })
        print("Training Finished !!!")
        

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_track = 0
        acc_track = 0
        for x, y in tqdm(self.trainloader, desc=f"Epoch {epoch+1}, Training "):
        # for x, y in self.trainloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                out = self.model(x, y, teacher_forcing_p=self.config.teacher_forcing_p, beam_size=1)
                loss, acc, _, _ = out
            
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
            # for x, y in self.validloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                    out = self.model(x, y, teacher_forcing_p=0.0, beam_size=1)
                    loss, acc, _, _ = out
                valid_loss_track += loss.item()
                valid_acc_track += acc.item()
            
            avg_valid_loss = valid_loss_track/len(self.validloader)
            avg_valid_acc = valid_acc_track/len(self.validloader)
        return avg_valid_loss, avg_valid_acc
    

    def inference(self, plot_attention=False):
        print("Starting inference ...")
        self.model.eval()
        with torch.no_grad():
            all_refs = []
            all_hyps = []
            for x, y in tqdm(self.testloader):
                x = x.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                    out = self.model(x, teacher_forcing_p=0, beam_size=self.config.beam_size)
                    _, _, decoder_outputs, _ = out

                # print("inference : ", decoder_outputs.shape)
                decoder_outputs = decoder_outputs.detach().cpu()
                y = y.cpu()
                
                if decoder_outputs.dim() == 3:
                    decoder_outputs_probs = F.log_softmax(decoder_outputs, dim=-1)
                    decoder_outputs = torch.argmax(decoder_outputs_probs, dim=-1)
                    

                all_refs.extend(list(y))
                all_hyps.extend(list(decoder_outputs))

            bleu_, targets, preds = self.scoring(all_refs, all_hyps, self.testloader.dataset)
            
            count = 0
            for t, p in zip(targets, preds):
                if t == p:
                    count+=1
            acc = count / len(targets) * 100
                    
            print(f"Test Accuracy : {acc:.4f}")

            idx = list(range(len(targets)))
            random.shuffle(idx)
            
            print("BLEU score:", bleu_)
            if self.logging:
                wandb.log({"bleu_score": bleu_, "test_accuracy": acc})
                
            idxs = random.sample(range(len(targets)+1), 9)

            if plot_attention:
                samples = []
                for idx in idxs:
                    sample = self.testloader.dataset.source.itos(self.testloader.dataset[idx][0])
                    samples.append(sample)
                print(samples)
                    
                self.predict_samples(samples, True)
            
            with open("with_attention_results.txt", "w", encoding="utf-8") as f:
                f.write("TARGETS\tPREDICTIONS\n\n")
                f.write("\n".join([f"{t}\t{p}" for t, p in zip(targets, preds)]))
                
                
 

    def scoring(self, refs, hyps, testset):
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


    def predict_samples(self, source_words, visualize_attention=False):
        attention_maps_list = []
        self.model.eval()
        with torch.no_grad():
            for source_word in source_words:
                s_idxs = self.trainloader.dataset.source.stoi(source_word)
                x = torch.tensor(s_idxs, dtype=torch.long).unsqueeze(0).to(self.device)
                
                out = self.model(x, teacher_forcing_p=0, beam_size=1, return_attention_map=visualize_attention)
                _, _, decoder_outputs, attention_map = out
                
                
                decoder_outputs = decoder_outputs.detach().cpu()
                probs = F.log_softmax(decoder_outputs, dim=-1)
                max_ = torch.argmax(probs, dim=-1)[0]
                eos_idx_hyp = (max_ == self.trainloader.dataset.target.EOS).nonzero(as_tuple=True)[0]
                hyp_trimmed = max_[:eos_idx_hyp[0].item()] if len(eos_idx_hyp) > 0 else max_
                target_word = self.trainloader.dataset.target.itos(hyp_trimmed)
                
                attention_map = attention_map.squeeze(1, 2)[:, :-1]
                attention_map = attention_map[:eos_idx_hyp[0].item()] if len(eos_idx_hyp) > 0 else attention_map
                attention_maps_list.append({"attention_map":attention_map.detach().cpu().numpy(), "source_word":source_word.replace('</s>', ''), "target_word":target_word})
            
            if visualize_attention:
                self.save_attention_map(attention_maps_list)
                make_q6(attention_maps_list)
                

    def save_attention_map(self, attn_data_list, save_path="attention_maps.png", figsize=(16, 16)):
        assert len(attn_data_list) == 9, "Expected exactly 9 samples for a 3x3 grid"

        hindi_font = font_manager.FontProperties(fname=nirm)

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        for i, data in enumerate(attn_data_list):
            attn = data["attention_map"]
            source = data["source_word"]
            target = data["target_word"]

            ax = axes[i]
            im = ax.imshow(attn, aspect='auto', cmap='viridis')

            ax.set_xticks(np.arange(len(source)))
            ax.set_xticklabels(list(source), fontsize=8, fontproperties=hindi_font, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(target)))
            ax.set_yticklabels(list(target), fontsize=8, fontproperties=hindi_font)

            ax.set_title(f"Tgt: '{target}' | Src: '{source.replace('</s>', '')}", fontsize=10, fontproperties=hindi_font)

        for j in range(len(attn_data_list), 9):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.015, pad=0.04)
        plt.savefig(save_path)
        plt.close()

        if self.logging:
            wandb.log({"attention_maps_grid": wandb.Image(save_path, caption="3x3 grid of attention maps")})

