# code source: https://www.kaggle.com/parthplc/finetune-mt5-for-question-generation-in-hindi
import argparse,os,logging,random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, random
import pytorch_lightning as pl
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# definition of T5 fine-tuner
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.model = MT5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,using_native_amp=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            logger.info("***** Validation results *****")
            if pl_module.is_logger():
                  metrics = trainer.callback_metrics
                  # Log results
                  for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                      logger.info("{} = {}\n".format(key, str(metrics[key])))

        def on_test_end(self, trainer, pl_module):
            logger.info("***** Test results *****")

            if pl_module.is_logger():
                metrics = trainer.callback_metrics

                  # Log and save results to file
                output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
                with open(output_test_results_file, "w") as writer:
                    for key in sorted(metrics):
                          if key not in ["log", "progress_bar"]:
                            logger.info("{} = {}\n".format(key, str(metrics[key])))
                            writer.write("{} = {}\n".format(key, str(metrics[key])))

class HyperGenerationDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=None):
        self.path = os.path.join(data_dir, type_path + '_hH.csv')

        self.source = 'source_text'
        self.target = 'target_text'
        self.data = pd.read_csv(self.path)

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_text,output_text= self.data.loc[idx, self.source],self.data.loc[idx, self.target]
   
            input_ = "INPUT: %s" % (input_text) #TODO
            target = "%s " %(output_text)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

def get_dataset(tokenizer, type_path, args):
    return HyperGenerationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


###################################################################################################################
# hyperparameters
args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='google/mt5-base',
    tokenizer_name_or_path='google/mt5-base',
    max_seq_length=64,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=8,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    use_multiprocessing=False,
)

# train_path = "../data/train_hH.csv"
# val_path = "../data/test_hH.csv"
# train = pd.read_csv(train_path)
# print (train.head())

# tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
# dataset = HyperGenerationDataset(tokenizer, '../data', 'train', 30)
# print("Val dataset: ",len(dataset))

# data = dataset[20]
# print(tokenizer.decode(data['source_ids']))
# print(tokenizer.decode(data['target_ids']))

args_dict.update({'data_dir': '../data', 'output_dir': 'output/', 'num_train_epochs':3,'max_seq_length':64})
args = argparse.Namespace(**args_dict)
print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    period =1,filepath=args.output_dir,  monitor="val_loss", mode="min", save_top_k=1, prefix="checkpoint"
)


train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()]
)


print ("Initialize model")
model = T5FineTuner(args)

print(torch.cuda.is_available())
trainer = pl.Trainer(**train_params)

print (" Training model")
trainer.fit(model)

print ("training finished")

print ("Saving model")
model.model.save_pretrained("/result")
print ("Saved model")
