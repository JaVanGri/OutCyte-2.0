import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import Bio.SeqIO as sio
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import esm

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_path = os.path.realpath(__file__)

# Erhalte den Verzeichnispfad zum aktuellen Skript
script_dir = os.path.dirname(script_path)

# Setze das Arbeitsverzeichnis auf das Verzeichnis des Skripts
#os.chdir(script_dir)


class OneHiddenLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(OneHiddenLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


def reduce_list(list, dimensions):
    return [list[i] for i in dimensions]


def read_fasta(filepath):
    records = sio.parse(filepath, 'fasta')
    seqs = [(str(r.id), str(r.seq)) for r in records if len(r.seq) >= 20]
    return pd.DataFrame(seqs, columns=['Entry', 'Sequence'])


def get_representations(df, model, alphabet, rep_layer, device):
    embedding = alphabet.get_batch_converter()
    representations = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for entry, seq in tqdm(df[['Entry', 'Sequence']].values, desc="Calculating representations"):
            batch_labels, batch_strs, batch_tokens = embedding([(entry, seq)])
            batch_tokens = batch_tokens.to(device)
            output = model(batch_tokens, repr_layers=[rep_layer])
            representation = output['representations'][rep_layer][:, 0, :].squeeze().cpu().numpy()
            representations.append(representation)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return representations


def load_model_parameters():
    with open(script_dir+'/parameter_upsv2/model_dimensions.json', 'r') as file:
        return json.load(file)


def load_model(model_key, rep_size, hidden_size):
    model = OneHiddenLayer(rep_size, hidden_size)
    model.load_state_dict(torch.load(script_dir+f'/parameter_upsv2/model_{model_key}.pth'))
    return model


def predict_proteins(df, model_dimensions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = {entry: [] for entry in df.Entry.tolist()}

    total_steps = len(df) * len(model_dimensions.keys())
    progress_bar = tqdm(total=total_steps,
                        desc=f'Calculating predictions from {len(model_dimensions.keys())} models for every protein')
    for model_key, dimensions in model_dimensions.items():
        model = load_model(model_key, 560, 40).to(device)
        model.eval()

        df['Representation_reduction'] = df['Representation'].apply(lambda x: reduce_list(x, dimensions))
        for entry, repr in df[['Entry', 'Representation_reduction']].values:
            t_repr = torch.Tensor([repr])
            t_repr = t_repr.to(device)
            with torch.no_grad():
                output = model.forward(t_repr)
                output = output.cpu()
                ups_prob = output[0][1].item()
                predictions[entry].append(ups_prob)

            progress_bar.update(1)

    for entry in df['Entry'].values:
        predictions[entry] = np.mean(predictions[entry])

    ups_prob = list(predictions.values())
    keys = list(predictions.keys())

    return pd.DataFrame({'entry':keys,'ups':ups_prob,'intracellular':1-np.array(ups_prob)})


def exclude_long_proteins(df, max_sequence_length):
    df['Length'] = df['Sequence'].str.len()
    df = df[df['Length'] <= max_sequence_length]
    df = df.sort_values(by='Length',ascending=False)
    df.index = range(len(df))
    return df


def predict_ups(entrys, sequences, device='cpu'):
    #Calculate Representations

    df = pd.DataFrame({'Entry':entrys,'Sequence':sequences})
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    df['Representation'] = get_representations(df, model, alphabet, rep_layer=30, device=device)

    #Calculate Predictions
    model_dimensions = load_model_parameters()
    predictions = predict_proteins(df, model_dimensions)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions

