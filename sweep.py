import wandb

data_path = "/speech/shoutrik/Databases/dakshina_dataset_v1.0"
language = "hindi"

sweep_config = {
    "method": "bayes",
    "project": "dakshina-transliteration",
    "entity": "shoutrik",
    "name": f"{language}-sweep",
    "metric": {
        "name": "valid_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "embedding_size": {"values": [32, 64, 256]},
        "encoder_num_layers": {"values": [1, 2, 3]},
        "decoder_num_layers": {"values": [1, 2, 3]},
        "hidden_size": {"values": [64, 256]},
        "encoder_name": {"values": ["RNN", "GRU"]},
        "decoder_name": {"values": ["RNN", "GRU"]},
        "dropout_p": {"values": [0.2, 0.3]},
        "learning_rate": {"values":[0.0001, 0.001, 0.003]},
        "teacher_forcing_p": {"values":[0.5, 0.8]},
        "apply_attention": {"values":[True, False]},
        "encoder_bidirectional": {"values":[True, False]},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"], entity=sweep_config["entity"])
print(f"Sweep created: {sweep_id}")

def sweep_run():
    from translit_trainer import Trainer, TrainerConfig

    wandb.init()
    config = wandb.config

    wandb.run.name = f"lr_{config.learning_rate}_hdsz_{config.hidden_size}_emb_{config.embedding_size}_enc_layers_{config.encoder_num_layers}_dec_layers_{config.decoder_num_layers}_enc_name_{config.encoder_name}_dec_name_{config.decoder_name}_enc_bidirectional_{config.encoder_bidirectional}_dropout_p_{config.dropout_p}_tfp_{config.teacher_forcing_p}_attn_{config.apply_attention}"

    trainer_config = TrainerConfig(
        language = language,
        data_path = data_path,
        learning_rate = config.learning_rate,
        teacher_forcing_p = config.teacher_forcing_p,
        embedding_size = config.embedding_size,
        hidden_size = config.hidden_size,
        encoder_num_layers = config.encoder_num_layers,
        decoder_num_layers = config.decoder_num_layers,
        encoder_name = config.encoder_name,
        decoder_name = config.decoder_name,
        encoder_bidirectional = config.encoder_bidirectional,
        dropout_p = config.dropout_p,
        apply_attention = config.apply_attention,
        batch_size = 64,
        num_workers = 16,
        weight_decay = 0.0005,
        decoder_SOS = 0,
    )
    
    
    trainer = Trainer(trainer_config)
    trainer.train()
    trainer.inference()
    wandb.finish()

wandb.agent(sweep_id, function=sweep_run, count=100)
