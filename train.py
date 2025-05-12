from translit_trainer import Trainer, TrainerConfig
from data_utils import TranslitDataset, preprocessor
import os

lang_map = {"bengali":"bn",
            "gujarati":"gu",
            "hindi":"hi",
            "kannada":"kn",
            "malayalam":"ml",
            "marathi":"mr",
            "punjabi":"pa",
            "sindhi":"sd",
            "sinhala":"si",
            "tamil":"ta",
            "telugu":"te",
            "urdu":"ur"}

language = "bengali"
data_path = "/speech/shoutrik/torch_exp/FliptyScripty/dakshina_dataset_v1.0/"

                   
train_data_path = os.path.join(data_path, f"{lang_map[language]}/lexicons/{lang_map[language]}.translit.sampled.train.tsv")
dev_data_path = os.path.join(data_path, f"{lang_map[language]}/lexicons/{lang_map[language]}.translit.sampled.dev.tsv")
test_data_path = os.path.join(data_path, f"{lang_map[language]}/lexicons/{lang_map[language]}.translit.sampled.test.tsv")

preprocessor(train_data_path, dev_data_path)
trainset = TranslitDataset(train_data_path, normalize=True)
validset = TranslitDataset(dev_data_path, normalize=False)
testset = TranslitDataset(test_data_path, normalize=False)


config = TrainerConfig(
    trainset = trainset,
    validset = validset,
    batch_size = 256,
    num_workers = 16,
    learning_rate = 0.003,
    weight_decay = 0.0005,
    teacher_forcing_p = 0.8,
    max_epoch = 10,
    embedding_size=256,
    hidden_size=256,
    encoder_num_layers=3,
    decoder_num_layers=2,
    encoder_name="GRU",
    decoder_name="GRU",
    encoder_bidirectional=True,
    dropout_p=0.3,
    max_length=32,
    apply_attention=True,
)


trainer = Trainer(config)
trainer.train()
trainer.inference(testset)





