# 🔤 Transliteration Model using Encoder-Decoder with Attention

This repository implements a **sequence-to-sequence transliteration model** using **RNN, GRU, or LSTM** based encoder-decoder architecture with **Bahdanau Attention**. It is designed for the **Dakshina dataset**, offering support for beam search decoding, attention map visualization, and Weights & Biases (wandb) logging for experiment tracking.

---

## 📌 Features

* Configurable Encoder-Decoder architecture (RNN / LSTM / GRU)
* Bahdanau Attention mechanism
* Beam Search decoding
* Attention map visualization with Hindi/Indic script support
* CLI configuration for complete training pipeline
* Weights & Biases (wandb) logging integration

---

## 📂 Dataset Structure

Use the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) and structure it as:


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shoutrix/TransliterationModel.git
cd TransliterationModel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Training the Model

Use the following command to train the model:

```bash
python run.py \
  --language hindi \
  --data_path /path/to/dakshina_dataset \
  --encoder_name GRU \
  --decoder_name LSTM \
  --beam_size 5 \
  --apply_attention True \
  --wandb_project Transliteration \
  --wandb_entity your_wandb_username
```

---

## ⚙️ Command-Line Arguments

| Argument                  | Description                                 | Default    |
| ------------------------- | ------------------------------------------- | ---------- |
| `--language`              | Language to train on                        | `hindi`    |
| `--data_path`             | Path to the Dakshina dataset                | *Required* |
| `--batch_size`            | Batch size for training                     | `256`      |
| `--num_workers`           | Number of data loader workers               | `16`       |
| `--learning_rate`         | Learning rate                               | `0.003`    |
| `--weight_decay`          | Weight decay                                | `0.0005`   |
| `--teacher_forcing_p`     | Teacher forcing probability                 | `0.8`      |
| `--max_epoch`             | Total training epochs                       | `10`       |
| `--embedding_size`        | Size of embedding layer                     | `256`      |
| `--hidden_size`           | Hidden state size for encoder and decoder   | `256`      |
| `--encoder_num_layers`    | Number of encoder layers                    | `1`        |
| `--decoder_num_layers`    | Number of decoder layers                    | `1`        |
| `--encoder_name`          | RNN type for encoder (`GRU`, `LSTM`, `RNN`) | `GRU`      |
| `--decoder_name`          | RNN type for decoder (`GRU`, `LSTM`, `RNN`) | `GRU`      |
| `--encoder_bidirectional` | Use bidirectional encoder                   | `True`     |
| `--dropout_p`             | Dropout probability                         | `0.3`      |
| `--max_length`            | Max decoding length                         | `32`       |
| `--apply_attention`       | Use Bahdanau attention                      | `True`     |
| `--beam_size`             | Beam size for decoding                      | `3`        |
| `--wandb_entity`          | wandb username                              | `None`     |
| `--wandb_project`         | wandb project name                          | `None`     |

---

## 📊 Logging with Weights & Biases

1. Create an account at [wandb.ai](https://wandb.ai/)
2. Set up your project and entity
3. Run the training with `--wandb_project` and `--wandb_entity`
4. (Optional) Set your `WANDB_API_KEY` as an environment variable for auto-login

---

## 🎯 Example Command

```bash
python train.py \
  --language hindi \
  --data_path /datasets/dakshina \
  --embedding_size 256 \
  --hidden_size 512 \
  --encoder_name LSTM \
  --decoder_name LSTM \
  --beam_size 4 \
  --dropout_p 0.2 \
  --wandb_project Transliteration \
  --wandb_entity mywandbuser
```

---

## 🖼️ Attention Map Visualization

If `plot_attention=True` is set during inference, attention maps will be saved as images and optionally logged to wandb. Hindi and other Indic scripts are rendered with custom fonts for clarity.

---

## 📥 Sample Predictions

You can optionally use the `predict_samples` method to visualize predictions and attention weights for selected input words:

```python
trainer.predict_samples(["dampattiyon", "dikhayi"], visualize_attention=True)
```

---
