import argparse
from translit_trainer import Trainer, TrainerConfig

def main(args):
    config = TrainerConfig(**vars(args))
    trainer = Trainer(config)
    trainer.train()
    trainer.inference()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transliteration model with Dakshina dataset")

    parser.add_argument("--language", type=str, default="hindi", help="Language to train on")
    parser.add_argument("--data_path", type=str, required=True, help="Root path of the Dakshina dataset")

    parser.add_argument("--batch_size", type=int, default=32)
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

    args = parser.parse_args()
    main(args)
