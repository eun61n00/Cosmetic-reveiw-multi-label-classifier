# !pip install mxnet-cu101
# !pip install gluonnlp pandas tqdm
# !pip install sentencepiece==0.1.85
# !pip install transformers==2.1.1
# !pip install torch==1.3.1
# !pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, random
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn import model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import gluonnlp as nlp

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model



# define hyperparameters
RANDOM_SEED = 77
USE_CUDA = torch.cuda.is_available()
OPTIMIZER_PARAM = {'lr': 1}
LEARNING_RATE = 5e-5
BATCH_SIZE = 50
EPOCH_MAX = 1000
EPOCH_LOG = 50
DROPOUT_RATE = 0.5
TOKENIZER_MAX_LEN = 140
WARMUP_RATIO = 0.1


# Prepare data input for BERT model
class Data_for_BERT(Dataset):
    def __init__(self, dataset, max_len, pad=True, pair=False, label_cols=label_cols):

        transform = nlp.data.BERTSentenceTransform(tokenizer=tok,
                                           max_seq_length=140,
                                           pad=True,
                                           pair=False)
        self.texts = dataset.text
        self.inputs = [transform([txt]) for txt in dataset.text]
        self.labels=dataset[label_cols].values

    def __getitem__(self, index):
        return (self.inputs[index] + (self.labels[index],))

    def __len__(self):

        return(len(self.labels))

class ReviewClassifier(nn.Module):
    def __init__(self, n_labels, bert_model, dr_rate = DROPOUT_RATE):
        super(ReviewClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout()
        self.out = nn.Linear(768, n_labels)
        # self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, token_ids, valid_length, segment_ids):
        _, output1 = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long())
        output2 = self.dropout(output1)
        output = self.out(output2)
        return output


def colwise_accuracy(y_true,y_pred):
    y_pred=y_pred.T
    y_true=y_true.T
    acc_list=[]
    for cate in range(0,y_pred.shape[0]):
        acc_list.append(accuracy_score(y_pred[cate],y_true[cate]))
    return sum(acc_list)/len(acc_list)

def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred > threshold, dtype=float)

    return {'Accuracy': accuracy_score(y_true=target, y_pred=pred),
            'Column-wise Accuracy': colwise_accuracy(y_true=target, y_pred=pred),
            'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def train(model, train_dataloader, loss_fn, optimizer, scheduler):

    path = os.getcwd() + '/drive/MyDrive/Colab Notebooks/project/bert_model_v2.pt'
    n_epochs = EPOCH_MAX

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    model.train()
    train_loss = []
    # train_epoch_pred=[]
    # train_loss_record=[]

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length

        label = label.float().to(device)

        out= model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

    return np.average(train_loss)



def evaluate(model, test_dataloader, loss_fn):

    model.eval()
    test_loss, n_correct, n_data = [], 0, 0
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, test_label) in enumerate(test_dataloader):

            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length

            test_label = test_label.float().to(device)

            test_out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(test_out, test_label)
            test_loss.append(loss.item())

            test_pred=test_out.detach().cpu().numpy()
            test_real=test_label.detach().cpu().numpy()

            test_batch_result = calculate_metrics(np.array(test_pred), np.array(test_real))

            test_accuracy = test_batch_result['Column-wise Accuracy']

        return np.average(test_loss), test_accuracy


def get_prediction_from_txt(input_text, threshold=0.0):
    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length = 140, pad=True, pair=False)
    sentences = transform([input_text])

    get_pred=model(torch.tensor(sentences[0]).long().unsqueeze(0).to(device),torch.tensor(sentences[1]).unsqueeze(0),torch.tensor(sentences[2]).to(device))
    pred=np.array(get_pred.to("cpu").detach().numpy()[0] > threshold, dtype=float)
    pred=np.nonzero(pred)[0].tolist()
    print(f"분석 결과, 대화의 예상 태그는 {[label_cases_sorted_target[i] for i in pred]} 입니다.")

    true=np.nonzero(input_text_label)[0].tolist()
    print(f"실제 태그는 {[label_cases_sorted_target[i] for i in true]} 입니다.")


if __name__ == '__main__':
    # 0. Preparation
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    print(f'device using: {device}')

    # 1.1 Load the dataset
    prj_path = os.getcwd() + '/drive/MyDrive/Colab Notebooks/project'
    df = pd.read_csv(f'{prj_path}/data/prd_review_df.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.dropna(inplace=True)
    df.drop(columns=['트러블 개선', '디자인', '거품', '용량', '품질', '향기', '보습력', '클렌징'], inplace=True)
    label_cols = list(df.columns[4:])

    # 1.2 Prepare model, tokenizer, vocabulary
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    # 1.3 Split and load training, valid dataset
    train_input, test_input, train_target, test_target = \
	    model_selection.train_test_split(df, np.array(df[label_cols]), test_size = 0.25, random_state = 42)

    train_input=train_input.reset_index(drop=True)
    test_input=test_input.reset_index(drop=True)

    data_train = Data_for_BERT(train_input, TOKENIZER_MAX_LEN, True, False, label_cols=label_cols)
    data_test = Data_for_BERT(test_input, TOKENIZER_MAX_LEN, True, False, label_cols=label_cols)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, num_workers=0)

    # 2. Instantiate a model, loss function, optimizer and scheduler
    model = ReviewClassifier(n_labels=len(label_cols), bert_model = bertmodel, dr_rate = DROPOUT_RATE).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_parameters = [
                            {
                                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                                "weight_decay": 0.001,
                            },
                            {
                                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                "weight_decay": 0.0,
                            },
                            ]
    optimizer = AdamW(optimizer_parameters, lr=LEARNING_RATE)
    loss_fn=nn.BCEWithLogitsLoss()

    warmup_step = int(len(train_dataloader)*EPOCH_MAX * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=len(train_dataloader)*EPOCH)

    # 3. Train the model
    train_loss_list, valid_loss_list = [], []
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, scheduler)
        valid_loss, valid_accuracy = evaluate(model, test_dataloader, loss_fn)

        train_loss_list.append([epoch, train_loss])
        valid_loss_list.append([epoch, valid_loss, valid_accuracy])
        if epoch % EPOCH_LOG == 0:
            print(f'{epoch:>6} TrLoss={train_loss:.6f}, VaLoss={valid_loss:.6f}, VaAcc={valid_accuracy: .3f}')

    # 4. Test
    input_text_num=random.randint(0, len(df))
    input_text=df.iloc[input_text_num,0]
    input_text=test_input.iloc[input_text_num,0]
    input_text_label=df.iloc[input_text_num,4:].tolist()

    print(input_text)
    get_prediction_from_txt(input_text, 0.0)
