from torch.utils.data import TensorDataset, DataLoader

from bin.scripts.metrics import calc_metrics

import pandas as pd
import torch
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def reducte_list(liste,dimensions):

    return [liste[i] for i in dimensions]

def data_to_table(data):
    df   = data['Representation'].tolist()
    df = [torch.tensor(tensor).numpy().flatten() for tensor in df]

    y = data['Label']
    X = pd.DataFrame(df)

    return X,y

def calc_best_dimensions_for_every_split(df, rep_c,):

    representation_count = rep_c
    X,y = data_to_table(df)

    estimator = LogisticRegression(max_iter=500)
    selector = RFE(estimator, n_features_to_select=representation_count, step=40)
    selector = selector.fit(X, y)
    list_of_chosen_dims=X.columns[selector.support_]
    return list_of_chosen_dims

def calc_embeddings(sequences, alphabet):
    embedding= alphabet.get_batch_converter()

    input = [('',seq) for seq in sequences]
    entry ,_, batch_tokens = embedding(input)

    return batch_tokens.long().tolist()

def calc_cls_representation(sequence,model,alphabet,rep_layer,device):
    embedding= alphabet.get_batch_converter()
    entry ,_, batch_tokens = embedding([('', sequence)])

    model= model.to(device)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        representation = model(batch_tokens,repr_layers=[rep_layer],return_contacts=False)['representations'][rep_layer][:,0,:]

    model = model.to('cpu')
    torch.cuda.empty_cache()
    return  representation.to('cpu')[0]

def get_dataloader(representations,label,batch_size):
    features = torch.tensor(representations).float()
    labels = torch.tensor(label).long()
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    return dataloader

def get_embedding_dataloader(representations,label,batch_size):
    features = torch.tensor(representations).long()
    labels = torch.tensor(label).long()
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    return dataloader

def train_step(model, dataloader, optimizer, criterion, device):
    model.train()  
    model.to(device)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device) 

        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step()  



def validation_step(model, dataloader, criterion, device):
    model.eval()  
    total_loss = 0
    total_correct = 0
    total = 0
    tp = 0  
    fp = 0  
    fn = 0  
    tn = 0  
    model.to(device)
    with torch.no_grad(): 
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device) 

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            for label, prediction in zip(labels.cpu(), predicted.cpu()):
                if label == 1 and prediction == 1:
                    tp += 1
                elif label == 1 and prediction == 0:
                    fn += 1
                elif label == 0 and prediction == 1:
                    fp += 1
                elif label == 0 and prediction == 0:
                    tn += 1

            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    model.to('cpu')
    return avg_loss, calc_metrics(tp, tn, fp, fn)


def prediction_step(model,alphabet, sequence, device):
    model.eval() 
    model.to(device)

    batch_converter = alphabet.get_batch_converter()
    data = [("",sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_tokens=batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens)

    results = results.to('cpu')

    return results[0][1].item()

def prediction_step_rep(model, representation, device):
    model.eval() 
    model.to(device)


    rep = torch.tensor(representation)
    rep = rep.to(device)
    rep = rep.unsqueeze(0)
  

    with torch.no_grad():
        results = model(rep)
        

    results = results.to('cpu')
    rep = rep.to('cpu')

    return results[0][1].item()

