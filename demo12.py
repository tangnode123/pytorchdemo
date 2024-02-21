#RNN名字识别
import csv
import gzip
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset,DataLoader

USE_GPU = False
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
# 数据准备
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'dataset/names_train.csv.gz' if is_train_set else 'dataset/names_test.csv.gz'
        with gzip.open(filename,"rt") as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len=len(self.names)
        self.counties = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.counties)))
        self.country_dict=self.getCountryDict()
        self.country_num=len(self.country_list)
    def __getitem__(self, item):
        return self.names[item],self.country_dict[self.counties[item]]
    def __len__(self):
        return self.len
    def getCountryDict(self):
        country_dict=dict()
        for i,country in enumerate(self.country_list,0):
            country_dict[country]=i
        return country_dict
    def idx2country(self,idx):
        return self.country_list[idx]

    def getCountriesNum(self):
        return self.country_num

trainset=NameDataset(is_train_set=True)
trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
testset=NameDataset(is_train_set=False)
testloader=DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)

N_COUNTRY=trainset.getCountriesNum()

def name2list(name):
    arr=[ord(c) for c in name]
    return arr,len(arr)
def create_tensor(tensor):
    if USE_GPU:
        device=torch.device("cuda:0")
        tensor=tensor.to(device)
    return tensor
def make_tensors(names,countries):
    sequence_and_lengths=[name2list(name) for name in names]
    name_squences=[s1[0] for s1 in sequence_and_lengths]
    squence_lengths=torch.LongTensor([s1[1] for s1 in sequence_and_lengths])
    countries=countries.long()

    seq_tensor=torch.zeros(len(name_squences),squence_lengths.max()).long()
    for idx,(seq,seq_len) in enumerate(zip(name_squences,squence_lengths),0):
        seq_tensor[idx,:seq_len]=torch.LongTensor(seq)

    seq_lengths,perm_idx=squence_lengths.sort(0,True)
    seq_tensor=seq_tensor[perm_idx]
    countries=countries[perm_idx]

    return create_tensor(seq_tensor),\
           create_tensor(seq_lengths),\
           create_tensor(countries)
class RNNNClassifier(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True):
        super(RNNNClassifier,self).__init__()
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.n_directions=2 if bidirectional else 1

        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)
        self.fc=nn.Linear(hidden_size*self.n_directions,output_size)

    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size)
        return create_tensor(hidden)

    def forward(self,input,seq_lengths):
        input=input.t()
        batch_size=input.size(1)

        hidden=self._init_hidden(batch_size)
        embedding=self.embedding(input)

        gru_input=pack_padded_sequence(embedding,seq_lengths)

        output,hidden=self.gru(gru_input,hidden)

        if self.n_directions == 2:
            hidden_cat=torch.cat([hidden[-1],hidden[-2]],dim=1)
        else:
            hidden_cat=hidden[-1]
        fc_output=self.fc(hidden_cat)
        return fc_output


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs ,seq_lengths,target = make_tensors(names,countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}',end='')
            print(f'[{i*len(inputs)}/{len(trainset)}]',end='')
            print(f'Loss: {total_loss / (i*len(inputs))}',end='\n')
    return total_loss

def tModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            predicted = output.max(dim=1,keepdim=True)[1]
            correct += predicted.eq(target.view_as(predicted)).sum().item()
        percent='%.2f'% (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':
    classifier=RNNNClassifier(N_CHARS,HIDDEN_SIZE,N_COUNTRY,N_LAYER)
    if USE_GPU:
        device=torch.device("cuda:0")
        classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer= optim.Adam(classifier.parameters(), lr= 0.001)

    start = time.time()
    print("Training for %d epochs.."%N_EPOCHS)
    acc_list = []
    for epoch in range(1,N_EPOCHS+1):
        trainModel()
        acc=tModel()
        acc_list.append(acc)