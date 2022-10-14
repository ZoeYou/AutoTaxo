import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os, re
from tqdm import tqdm

from transformers import (
        T5Tokenizer, 
        T5ForConditionalGeneration, 
        MT5Tokenizer, 
        MT5ForConditionalGeneration, 
        AutoTokenizer,
        AdamW
)


# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.empty_cache()


class SinoPatDataSetClass(Dataset):

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    with tqdm(enumerate(loader, 0), unit="batch", total = len(loader)) as tepoch:
        for _, data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            y = data["target_ids"].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())#, accuracy=100. * accuracy)

    training_logger.add_row(str(epoch), str(_), str(loss))
    console.print(training_logger)

def validate(epoch, tokenizer, model, device, loader, beam_size):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=128, 
              num_beams=beam_size+5,
              num_return_sequences=beam_size,
              repetition_penalty=1.1,   # The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
              length_penalty=0.0,   # Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length. 0.0 means no penalty. Set to values < 0.0 in order to encourage the model to generate longer sequences, to a value > 0.0 in order to encourage the model to produce shorter sequences.
              early_stopping=True   # generation is finished when all beam hypotheses reached the EOS token.
              )

          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend([p.replace("<extra_id_0>","").strip() for p in preds])
          actuals.extend(target)
  return predictions, actuals

def hit_at_k(y_true, y_pred, k=10):
    """Computing hit score metric at k.
    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    Returns:
        np.ndarray: hit score.
    """
    ground_truth = y_true
    predictions = y_pred[:k]   
    correct = list(set(ground_truth).intersection(set(predictions)))
    return len(correct)/len(ground_truth)

def MRR(y_true, y_pred):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = [[gold == pred for gold in y_true for pred in y_pred]]
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = MT5Tokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_tokens(['<A>','<B>','<C>','<D>','<E>','<F>','<G>','<H>'])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = SinoPatDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = SinoPatDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = AdamW(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader, model_params["BEAM_SIZE"])
        predictions = [predictions[x:x+model_params["BEAM_SIZE"]] for x in range(0, len(predictions), model_params["BEAM_SIZE"])]

        h1_scores = [hit_at_k([a], p, k=1) for p,a in zip(predictions, actuals)]
        h3_scores = [hit_at_k([a], p, k=3) for p,a in zip(predictions, actuals)]
        h10_scores = [hit_at_k([a], p, k=10) for p,a in zip(predictions, actuals)]
        mrrs = [MRR([a], p) for p,a in zip(predictions, actuals)]

        final_df = pd.DataFrame({"source_text": val_dataset["source_text"].tolist(), "Generated Text": predictions, "Actual Text": actuals, "H@1": h1_scores, "H@3": h3_scores, "H@10": h10_scores, "MRR": mrrs})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

# let's define model parameters specific to T5
model_params = {
    "MODEL": "google/mt5-base",  # model_type: t5-base/t5-large | google/mt5-small 
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 128,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 128,  # max length of target text
    "SEED": 666,  # set seed for reproducibility
    "BEAM_SIZE": 10, # number of beam search
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':

    def modify(line):
        domain, term = line.split(":")[1].split(">")  
        return "predict hypernym" + domain + ">: " + term 

    # load dataset
    df = pd.read_csv(args.data_file, sep='\t', names=["source_text", "target_text"])
    #df['source_text'] = df['source_text'].apply(modify)

    torch.cuda.empty_cache()
    T5Trainer(
        dataframe=df,
        source_text="source_text",
        target_text="target_text",
        model_params=model_params,
        output_dir=args.data_file.split("/")[-1].split(".")[-2]
    )










