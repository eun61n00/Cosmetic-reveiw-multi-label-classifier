import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import random
from sklearn import metrics, model_selection, preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


from metrics import *

prj_path = os.getcwd() + '/drive/MyDrive/Colab Notebooks/project'

df = pd.read_csv(f'{prj_path}/data/prd_review_df.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.dropna(inplace=True)


# define hyperparameters
RANDOM_SEED = 777
model_config={"max_len" :512,
              "batch_size":5,
             "warmup_ratio": 0.1,
             "num_epochs": 200,
             "max_grad_norm": 1,
             "learning_rate": 5e-6,
              "dr_rate":0.45}

label_cols = ['sentiment',
              '사용감',
              '트러블 개선',
              '디자인',
              '효과',
              '발림성',
              '거품',
              '발색',
              '만족도',
              '커버력',
              '용량',
              '품질',
              '향기',
              '보습력',
              '지속력',
              '색상',
              '가격',
              '클렌징']

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

tokenizer = get_tokenizer()
bertmodel, vocab = get_pytorch_kobert_model()

tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device using: {device}')

class Data_for_BERT(Dataset):
    def __init__(self, dataset, max_len, pad=True, pair=False, label_cols=label_cols):

        transform = nlp.data.BERTSentenceTransform(tokenizer=tok,
                                           max_seq_length=140,
                                           pad=True,
                                           pair=False)
        self.sentences = [transform([txt]) for txt in dataset.text]
        self.labels=dataset[label_cols].values

    def __getitem__(self,i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return(len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self, hidden_size = 768, num_classes = 8, dr_rate = None, params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bertmodel
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def generate_attention_mask(self, token_ids, valid_length):

        attention_mask = torch.zeros_like(token_ids)

        for i,v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.generate_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out=self.dropout(pooler)

        return self.classifier(out)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
	train_test_split(df[:30000], np.array(df[label_cols][:30000]), test_size = 0.25, random_state = 42)

train=train_input.copy()
test=test_input.copy()

train=train.reset_index(drop=True)
test=test.reset_index(drop=True)

data_train = Data_for_BERT(train, model_config["max_len"], True, False, label_cols=label_cols)
data_test = Data_for_BERT(test, model_config["max_len"], True, False, label_cols=label_cols)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=model_config["batch_size"], num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=model_config["batch_size"], num_workers=0)


# model, optimizer, scheduler, loss function
model = BERTClassifier(num_classes=18, dr_rate = model_config["dr_rate"]).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay' : 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr= model_config["learning_rate"])
loss_fn=nn.BCEWithLogitsLoss()

t_total = len(train_dataloader) * model_config["num_epochs"]
warmup_step = int(t_total * model_config["warmup_ratio"])
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# train
def train_model(model, batch_size, patience, n_epochs,path):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    for epoch in tqdm(range(1, n_epochs + 1)):

        # initialize the early_stopping object
        model.train()
        train_epoch_pred=[]
        train_loss_record=[]

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length

            # label = label.long().to(device)
            label = label.float().to(device)

            out= model(token_ids, valid_length, segment_ids)#.squeeze(1)

            loss = loss_fn(out, label)

            train_loss_record.append(loss)

            train_pred=out.detach().cpu().numpy()
            train_real=label.detach().cpu().numpy()

            train_batch_result = calculate_metrics(np.array(train_pred), np.array(train_real))

            if batch_id%50==0:
                print(f"batch number {batch_id}, train col-wise accuracy is : {train_batch_result['Column-wise Accuracy']}")


            # save prediction result for calculation of accuracy per batch
            train_epoch_pred.append(train_pred)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config["max_grad_norm"])
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            train_losses.append(loss.item())

        train_epoch_pred=np.concatenate(train_epoch_pred)
        train_epoch_target=train_dataloader.dataset.labels
        train_epoch_result=calculate_metrics(target=train_epoch_target, pred=train_epoch_pred)

        print(f"=====Training Report: mean loss is {sum(train_loss_record)/len(train_loss_record)}=====")
        print(train_epoch_result)

        print("=====train done!=====")

        # if e % log_interval == 0:
        #     print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))

        # print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        test_epoch_pred=[]
        test_loss_record=[]

        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, test_label) in enumerate(test_dataloader):

                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length

                # test_label = test_label.long().to(device)
                test_label = test_label.float().to(device)

                test_out = model(token_ids, valid_length, segment_ids)

                test_loss = loss_fn(test_out, test_label)

                test_loss_record.append(test_loss)

                valid_losses.append(test_loss.item())

                test_pred=test_out.detach().cpu().numpy()
                test_real=test_label.detach().cpu().numpy()

                test_batch_result = calculate_metrics(np.array(test_pred), np.array(test_real))

                if batch_id%50==0:
                    print(f"batch number {batch_id}, test col-wise accuracy is : {test_batch_result['Column-wise Accuracy']}")

                # save prediction result for calculation of accuracy per epoch
                test_epoch_pred.append(test_pred)

        test_epoch_pred=np.concatenate(test_epoch_pred)
        test_epoch_target=test_dataloader.dataset.labels
        test_epoch_result=calculate_metrics(target=test_epoch_target, pred=test_epoch_pred)

        print(f"=====Testing Report: mean loss is {sum(test_loss_record)/len(test_loss_record)}=====")
        print(test_epoch_result)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path))

    return  model, avg_train_losses, avg_valid_losses

weight_path = os.getcwd() + '/drive/MyDrive/Colab Notebooks/project/bert_model.pt'
model.load_state_dict(torch.load(weight_path))

# Training
patience = 10
model, train_loss, valid_loss = train_model(model,
                                            model_config["batch_size"],
                                            patience,
                                            model_config["num_epochs"],
                                            path=weight_path)