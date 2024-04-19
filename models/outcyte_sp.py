import os
import h5py
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable as var
import numpy as np
import pandas as pd
import sys
import re
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

def run_sp(seqID, seqs):
    """predictions for the input sequences"""
    predictions = []
    scores = []
    ohk = list()
    for seq in seqs:
        #print('seq in run_sp', seq)
        seq = string_replace(seq)
        ohk.append(gen_ohk_aas(seq))
    N = len(seqID)
    ohk = np.array(ohk)
    prob_all = []
    model_path = current_dir+'/parameter_sp/'
    for root, dirs, fls in os.walk(model_path):
        for name in dirs:
            abspath = model_path + name + '/'
            w_dict = load_weights(abspath)
            prob = model(
                ohk,
                N,
                w_dict
            )
            prob_all.append(prob)
    prob_all = np.array(prob_all)
    scores = np.mean(prob_all, axis=0)
    predictions = np.argmax(scores, axis=1)
    class_dict = {0:'Intracellular',
                  1: 'Signal-peptide',
                  2: 'Transmembrane'
                 }

    transmembrane = []
    signal_peptide = []
    intracellular = []
    for i in range(len(scores)):
        transmembrane.append(scores[i][2])
        signal_peptide.append(scores[i][1])
        intracellular.append(scores[i][0])


    results = pd.DataFrame({'entry':seqID,'transmembrane':transmembrane,'signal_peptide':signal_peptide,'intracellular':intracellular})
    return results

def read_fasta(inputs):
    res = re.findall(">.*?\n\r|(?:[^>].*?\n\r)+", inputs)
    ids = res[0::2]
    seqs = res[1::2]
    ids = [i.split(' ')[0].split('>')[1] if (' ' in i) == True else i[1:-1] for i in ids]
    seqs = [('').join(i.split('\n')) for i in seqs]
    return ids, seqs

def gen_ohk_aas(aas):
    name = ['G', 'P', 'A', 'V', 'L',
            'I', 'M', 'C', 'F', 'Y', 'W',
            'H', 'K', 'R', 'Q', 'N', 'E',
            'D', 'S', 'T']
    codes = np.eye(20)
    df = pd.DataFrame(codes, columns=name)
    length = len(aas)
    aas = str(aas.replace('\r',''))
    aas = ''.join(aas)
    if length >= 70:
        aas = aas[0:70]
        '''aas = re.findall(r'\w*', aas)[0]
        f = open('/home/rafiee/bioml/bioml/res.txt','w')
        aas_test = ""
        for item in aas:
            if item:
                aas_test += item
        f.writelines(aas_test)
        aas = aas_test
        f.close()'''
        hot_coding = []
        for i in range(70):
            hot_coding.append(df[aas[i]].values.tolist())
        ohk = np.array(hot_coding, dtype=np.float32)
    elif length < 70:
        hot_coding = []
        for i in range(length):
            hot_coding.append(df[aas[i]].values.tolist())
        ohk = np.array(hot_coding, dtype=np.float32)
        ohk = np.concatenate((ohk, np.zeros((70-length, 20))), axis=0)
    return ohk

def string_replace(text):
    '''
    Replace uncommon amino acids
    '''
    dic = {
    'X': 'Q',
    'Z': 'E',
    'B': 'N',
    'U': 'C',
    'J': 'L',
    'O': 'K'
    }

    for i, j in dic.items():
        text = text.replace(i, j)
    return text
def load_weights(path):
    '''
    Load trained model weights layerwise into a dict
    '''
    w_dict = {}
    for rt, fls, dt in os.walk(path):
        #print rt
        for fd in dt:
            #print fd
            fname = fd.split('.')[0]
            h5f = h5py.File(rt+fd, 'r')
            data = h5f[fname][:]
            #print(fname)
            #print(dataset.shape)
            w_dict[fname] = data
    return w_dict

def convert_kernel(kernel):
    """
    NOTE: This is adopted from Keras'datasets implementation where flips the first
    two dimensions, but in Theano and Pytorch, the actual kernel is the last
    two dimensions.
    Converts a Numpy kernel matrix from Theano format to TensorFlow format.
    Also works reciprocally, since the transformation is its own inverse.
    # Arguments
        kernel: Numpy array (3D, 4D or 5D).
    # Returns
        The converted kernel.
    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = np.asarray(kernel)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[:2] = no_flip
    return np.copy(kernel[slices])


def model(data, N, w_dict):
    #print(dataset.shape)
    inputs = np2var(data.reshape((N, 1, 70, 20))).float()
    layer0_w = np2var(convert_kernel(w_dict['layer0_w']))
    layer0_b = np2var(w_dict['layer0_b'])
    #print(layer0_w.shape, layer0_b.shape)
    layer1_w = np2var(convert_kernel(w_dict['layer1_w'])) #flip the kernels from Theao
    layer1_b = np2var(w_dict['layer1_b'])
    layer2_w = w_dict['layer2_w']
    layer2_b = w_dict['layer2_b']
    softmax_w = w_dict['softmax_w']
    softmax_b = w_dict['softmax_b']

    conv0_out = F.conv2d(input=inputs,  #(N, 1, 70, 20)
                         weight=layer0_w, #(5, 1, 1, 20)
                         bias=layer0_b  #(5, )
                        )
    layer0_out = np2var(relu(var2np(conv0_out))) #(N, 5, 70, 1)
    conv1_out = F.conv1d(input=layer0_out.squeeze(dim=3),
                         weight=layer1_w.squeeze(), #(15, 5, 30)
                         bias=layer1_b, #(15, )
                         padding=15
                        ) #(N, 15, 71, 1)
    #print(conv1_out.shape)
    layer1_out = np.max(relu(var2np(conv1_out)),axis=2).reshape(N, 15)
    dense_out=relu(np.matmul(layer1_out, layer2_w) + layer2_b)
    #sm_out = softmax(np.matmul(dense_out, softmax_w) + softmax_b)
    sm_out_t = F.softmax(np2var(np.matmul(dense_out, softmax_w) + softmax_b), dim=1)
    sm_out = var2np(sm_out_t)
    return sm_out

def relu(x):
    return x * (x > 0)
def np2var(x):
    return var(T.from_numpy(x))
    #return T.from_numpy(x) #for updated PyTorch version
def var2np(x):
    return x.data.cpu().numpy()
    #return x.numpy()
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.amin(x))
    return e_x / e_x.sum()

'''if __name__ == '__main__':
    #print(gen_ohk_aas('MSEALKILNNIRTLRAQARECTLETLEEMLEKLEVVVNERREEESAAAAEVEERTRKLQQYREMLIADGI'))
    res = run_sp(['1'], ['MSEALKILNNIRTLRAQARECTLETLEEMLEKLEVVVNERREEESAAAAEVEERTRKLQQYREMLIADGI'])
    print(res)'''
