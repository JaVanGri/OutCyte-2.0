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
import gc
import warnings
import tracemalloc
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_path = os.path.realpath(__file__)

script_dir = os.path.dirname(script_path)



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


def get_predictions(df, model, alphabet, rep_layer, device):
    model_dimensions = load_model_parameters()
    embedding = alphabet.get_batch_converter()
    predictions = {entry: [] for entry in df.Entry.tolist()}

    representation_model = model
    prediction_models = {key: load_model(key, 560, 40) for key in model_dimensions.keys()}

    if device == 'cuda':
        representation_model.to(device)
    representation_model.eval()

    for key in model_dimensions.keys():
        model = prediction_models[key]
        if device == 'cuda':
            model.to(device)
        model.eval()

    with torch.no_grad():
        for i, (entry, seq) in enumerate(tqdm(df[['Entry', 'Sequence']].values, desc="Calculating ups predictions...")):
            batch_labels, batch_strs, batch_tokens = embedding([(entry, seq)])
            batch_tokens = batch_tokens.to(device)
            output = representation_model(batch_tokens, repr_layers=[rep_layer])
            if device == 'cuda':
                representation = output['representations'][rep_layer][:, 0, :].squeeze().cpu().numpy()
            else:
                representation = output['representations'][rep_layer][:, 0, :].squeeze().numpy()

            for key in model_dimensions.keys():
                model = prediction_models[key]

                dimensions = model_dimensions[key]
                reduced_representation = reduce_list(representation, dimensions)

                t_repr = torch.Tensor([reduced_representation])
                t_repr = t_repr.to(device)
                with torch.no_grad():
                    output = model.forward(t_repr)
                    output = output.cpu()
                    ups_prob = output[0][1].item()
                    predictions[entry].append(ups_prob)

            predictions[entry] = np.mean(predictions[entry])


            del batch_tokens, output, batch_labels, batch_strs, representation
            if device == 'cuda':
                torch.cuda.empty_cache()
            if i % 100 == 0:
                gc.collect()

    ups_prob = list(predictions.values())
    keys = list(predictions.keys())

    return pd.DataFrame({'entry': keys, 'ups': ups_prob, 'intracellular': 1 - np.array(ups_prob)})

def load_model_parameters():
    with open(script_dir+'/parameter_upsv2/model_dimensions.json', 'r') as file:
        return json.load(file)


def load_model(model_key, rep_size, hidden_size):
    model = OneHiddenLayer(rep_size, hidden_size)
    model.load_state_dict(torch.load(script_dir+f'/parameter_upsv2/model_{model_key}.pth'))
    return model


def exclude_long_proteins(df, max_sequence_length):
    df['Length'] = df['Sequence'].str.len()
    df = df[df['Length'] <= max_sequence_length]
    df = df.sort_values(by='Length',ascending=False)
    df.index = range(len(df))
    return df


def predict_ups(entrys, sequences, device='cpu'):
    df = pd.DataFrame({'Entry':entrys,'Sequence':sequences})
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    return  get_predictions(df, model, alphabet, rep_layer=30, device=device)

