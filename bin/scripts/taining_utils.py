from torch.utils.data import TensorDataset, DataLoader

from bin.scripts.metrics import calc_metrics

import pandas as pd
import torch
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def reducte_list(liste,dimensions):
    #print(liste)
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

    estimator = LogisticRegression(max_iter=500)#SVR(kernel="linear")
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

    #print(batch_tokens)
    with torch.no_grad():
        representation = model(batch_tokens,repr_layers=[rep_layer],return_contacts=False)['representations'][rep_layer][:,0,:]

    #print(representation.shape)
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
    model.train()  # Setzt das Modell in den Trainingsmodus
    model.to(device)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Verschiebe Daten und Labels auf das richtige Gerät

        optimizer.zero_grad()  # Setzt die Gradienten der optimierten Tensor zurück
        outputs = model(inputs)  # Führt einen Vorwärtsdurchlauf durch
        loss = criterion(outputs, labels)  # Berechnet den Verlust
        loss.backward()  # Berechnet die Gradienten
        optimizer.step()  # Aktualisiert die Parameter



def validation_step(model, dataloader, criterion, device):
    model.eval()  # Setzt das Modell in den Evaluierungsmodus
    total_loss = 0
    total_correct = 0
    total = 0
    tp = 0  # Wahre Positive
    fp = 0  # Falsche Positive
    fn = 0  # Falsche Negative
    tn = 0  # Wahre Negative
    model.to(device)
    with torch.no_grad():  # Deaktiviert die Gradientenberechnung
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Verschiebe Daten und Labels auf das richtige Gerät

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            # Update der TP, FP, FN, TN Zähler
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
    model.eval()  # Setzt das Modell in den Evaluierungsmodus
    model.to(device)

    batch_converter = alphabet.get_batch_converter()
    data = [("",sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_tokens=batch_tokens.to(device)
    #print(batch_tokens.shape)
    with torch.no_grad():
        results = model(batch_tokens)
        #print(results[0][1],results[0][1].item())

    results = results.to('cpu')
    #print(results.item())
    return results[0][1].item()

def prediction_step_rep(model, representation, device):
    model.eval()  # Setzt das Modell in den Evaluierungsmodus
    model.to(device)


    rep = torch.tensor(representation)
    rep = rep.to(device)
    rep = rep.unsqueeze(0)
    #print(rep.shape)

    with torch.no_grad():
        results = model(rep)
        #print(results[0][1],results[0][1].item())

    results = results.to('cpu')
    rep = rep.to('cpu')
    #print(results.item())
    return results[0][1].item()

