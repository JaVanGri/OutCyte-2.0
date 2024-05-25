import argparse
import esm
import pandas as pd
import torch
import numpy as np
from bin.scripts.taining_utils import get_dataloader, calc_cls_representation, calc_best_dimensions_for_every_split, \
    reducte_list
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.utils import compute_class_weight
from bin.scripts.taining_utils import train_step, validation_step
import warnings
import json


warnings.filterwarnings("ignore", category=Warning)


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on sequence data.')
    parser.add_argument('training_data', type=str,
                        help='Path to the CSV file containing training data, needs columns Entry,Label and Sequence')
    return parser.parse_args()


# Calculate representations
def calculate_representations(data, model, alphabet, layer, device):
    reprs = []
    for sequence in tqdm(data['Sequence']):
        reprs.append(calc_cls_representation(sequence, model, alphabet, rep_layer=layer, device=device))
    data['Representation'] = reprs
    data['Representation'] = data['Representation'].apply(lambda x: [t.item() for t in x])
    return data


# Define model
class OneHiddenLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(OneHiddenLayer, self).__init__()

        self.p = dropout_p
        self.dropout = nn.Dropout(p=self.p)
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


# Training
def train(data, device, rep_size, hidden_size, drop_out, lr, weight_decay, epochs,
          batch_size, smp_size_pos, smp_size_neg, runs, patience=20, min_delta=0.001):

    model_dimensions = {}
    for i in tqdm(range(runs)):
        data_train, data_val = sample_data(data, smp_size_pos,smp_size_neg)
        dimensions = calc_best_dimensions_for_every_split(data_train, rep_size)
        model_dimensions[str(i)] = list(dimensions)

        data_train = reduce_representations(data_train, dimensions)
        data_val = reduce_representations(data_val, dimensions)

        dl_train = get_dataloader(data_train['Representation_reduction'].tolist(), data_train['Label'].tolist(),
                                  batch_size)
        dl_val = get_dataloader(data_val['Representation_reduction'].tolist(), data_val['Label'].tolist(), 1)

        class_weights = compute_class_weights(data_train, device)
        model = OneHiddenLayer(rep_size, hidden_size, dropout_p=drop_out).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_metric = None
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_step(model, dl_train, optimizer, criterion, device)
            v_loss, v_metrics = validation_step(model,dl_val,criterion,device)

            metric = v_metrics['f1']

            if best_metric is None or metric >= best_metric + min_delta:
                best_metric = metric
                epochs_no_improve = 0 
                torch.save(model.state_dict(), f'bin/models/parameter_upsv2/model_{i}.pth')

            else:
                epochs_no_improve += 1 

            if epochs_no_improve >= patience:

                break

    with open('bin/models/parameter_upsv2/model_dimensions.json', 'w') as file:
        json.dump(model_dimensions, file, indent=4, sort_keys=True)



def sample_data(data, smp_size_pos,smp_size_neg):
    data_pos_smp = data[data['Label'] == 1].sample(smp_size_pos)
    data_neg_smp = data[data['Label'] == 0].sample(smp_size_neg)
    data_val = pd.concat([data_pos_smp, data_neg_smp])
    data_train = data[~data['Entry'].isin(data_val['Entry'])]

    data_val.index = range(len(data_val))
    data_train.index = range(len(data_train))
    return data_train, data_val


def reduce_representations(data, dimensions):
    data['Representation_reduction'] = data['Representation'].apply(lambda x: reducte_list(x, dimensions))
    return data


def compute_class_weights(data_train, device):
    class_weights = compute_class_weight('balanced', classes=np.unique(data_train['Label']), y=data_train['Label'])
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def evaluate_model(model, dataloader, criterion, device):
    loss, metrics = validation_step(model, dataloader, criterion, device)
    metrics_df = pd.DataFrame({'accuracy': [metrics['accuracy']], 'sensitivity': [metrics['sensitivity']],
                               'specificity': [metrics['specificity']]})
    return metrics_df


def main():
    args = parse_args()

    data = pd.read_csv(args.training_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

    repr_layer = 30
    data = calculate_representations(data, model, alphabet, repr_layer, device)

    lr = 1e-3
    batch_size = 32
    weight_decay = 1e-3
    drop_out = 0.8
    hidden_size = 40
    rep_size = 560
    epochs = 100
    smp_size_pos = 17
    smp_size_neg = 90
    runs = 100

    train(data, device, rep_size, hidden_size, drop_out, lr, weight_decay,
          epochs, batch_size, smp_size_pos, smp_size_neg, runs)



if __name__ == "__main__":
    main()
