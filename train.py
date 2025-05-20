import argparse
from translit_trainer import Trainer, TrainerConfig
import wandb

def main(args, logging):
    args = vars(args)
    args = {k:v for k, v in args.items() if k not in ["wandb_entity", "wandb_project"]}
    config = TrainerConfig(**args, logging=logging)
    
    print("Config --->")
    print(config)
    
    trainer = Trainer(config, logging=logging)
    trainer.train()
    trainer.inference()
    # trainer.predict_one("tughlakabad", visualize_attention=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transliteration model with Dakshina dataset")
    
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, required=None)

    parser.add_argument("--language", type=str, default="hindi", help="Language to train on")
    parser.add_argument("--data_path", type=str, required=True, help="Root path of the Dakshina dataset")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--teacher_forcing_p", type=float, default=0.8)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--encoder_num_layers", type=int, default=3)
    parser.add_argument("--decoder_num_layers", type=int, default=2)
    parser.add_argument("--encoder_name", type=str, default="GRU", choices=["GRU", "LSTM", "RNN"])
    parser.add_argument("--decoder_name", type=str, default="GRU", choices=["GRU", "LSTM", "RNN"])
    parser.add_argument("--encoder_bidirectional", type=bool, default=True)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--apply_attention", type=bool, default=True)
    parser.add_argument("--beam_size", type=int, default=3)

    

    args = parser.parse_args()
    
    if args.wandb_entity is not None or args.wandb_project is not None:
        wandb.init(project=args.wandb_project)
        wandb.run.name = f"BestModelWithAttention_beam_{args.beam_size}_lr_{args.learning_rate}_hdsz_{args.hidden_size}_emb_{args.embedding_size}_enc_layers_{args.encoder_num_layers}_dec_layers_{args.decoder_num_layers}_enc_name_{args.encoder_name}_dec_name_{args.decoder_name}_enc_bidirectional_{args.encoder_bidirectional}_dropout_p_{args.dropout_p}_tfp_{args.teacher_forcing_p}_attn_{args.apply_attention}"
        logging=True
    else:
        logging=False
    
    main(args, logging)
