import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import DictVectorizer
import joblib
import xgboost
#import eli5
import pandas as pd
from collections import Counter
from Bio.SeqUtils import ProtParam
from Bio import SeqIO as sio 
import re
import multiprocessing
import concurrent.futures
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
count=0
SLC = 'ILVFMCAGPTSYWQNHEDKRU' #12, 3, ..., 19
FEATURE_NAMES = [
    'W',
    'F',
    'pos_freq', #1
    'mol_weight', #1
    'small_c',  #1
    'R',
    'hydrophobic_c', #1
    'positive_c', #1
]

def run_ups(seqID, seqs, show_contribution=False):
    '''
    run ups model on the input sequences.
    '''
    feature, ids = generator(seqID, seqs,para=False)
    #print('Calculated Features')
    #print('The id', ids)
    features = [np.array(feature[x]) if x=='aaf' \
        else np.expand_dims(np.array(feature[x]), axis=1)\
        for x in FEATURE_NAMES
        ]
    print(len(features[0]))
    features = np.concatenate(features, axis=1)
    #print(features)
    feature_dict = {k:feature[k] for k in FEATURE_NAMES}
    N = features.shape[0]
    #force the value to be false if inputs have more than one sequence
    #if N > 1:
    #    show_contribution = False
    #print('APPLY MODEL')
    val = apply_model(features, feature_dict)
    #    val = [predicted_y, predicted_idx, predicted_score, explanation]
    predicted_class = np.around(val[0], 4)
    class_dict = {
        0:'Intracellular',
        1:'UPS'
    }

    ups = []
    intracellular = []
    for i,u in val[2]:
        ups.append(u)
        intracellular.append(i)

    result = pd.DataFrame({'entry':ids,'ups':ups,'intracellular':intracellular})

    return result

def apply_model(feature, feature_dict, y_true=None):
    '''
    Apply the saved models to input sequence or fasta file.

    If the input labels are not given, return the predicted
    labels and corresponding indices.

    If the labels are given, return the metrics as well
    '''
    scores = []
    for root, dirs, files in os.walk(current_dir+'/parameter_ups/'):
        for fl in files:
            abspath = root+fl
            #print abspath
            model = joblib.load(abspath)
            predict_score = model.predict_proba(feature)
            scores.append(predict_score)
    '''if show_contribution:
        model_cont = joblib.load('/home/rafiee/bioml/outcyte/model_ups_contribution.pkl')
        eli5.xgboost._check_booster_args = _check_booster_args
        vec = DictVectorizer(sparse=False)
        feature = vec.fit_transform(feature_dict)
        #print('Feature contribution saved to file!')
        expl = eli5.explain_prediction(
            model_cont,
            feature_dict,
            vec=vec)
        explanation = eli5.format_as_text(expl)
    else: explanation = None
    '''
    predicted_score = np.mean(np.asarray(scores), axis=0)
    #print predicted_score
    #print(predicted_score)
    predicted_y = np.argmax(predicted_score, axis=1)
    predicted_idx = (predicted_y == 1).nonzero()[0]
    #predicted_score = model.predict_proba(feature)

    if y_true is not None:
        ACC = metrics.accuracy_score(y_true, predicted_y)
        F1 = metrics.f1_score(y_true, predicted_y)
        AUC = roc(y_true, predicted_score[:, 1])
        PPV = metrics.precision_score(y_true, predicted_y)
        TPR = metrics.recall_score(y_true, predicted_y)
        val = [predicted_y, predicted_idx, predicted_score, ACC, F1, AUC, PPV, TPR]
    else:
        val = [predicted_y, predicted_idx, predicted_score]
    return val

'''def _check_booster_args(xgb, is_regression=None):
    # type: (Any, bool) -> Tuple[Booster, bool]
    if isinstance(xgb, eli5.xgboost.Booster): # patch (from "xgb, Booster")
        booster = xgb
    else:
        booster = xgb.get_booster() # patch (from "xgb.booster()" where `booster` is now a string)
        _is_regression = isinstance(xgb, xgboost.XGBClassifier)
        if is_regression is not None and is_regression != _is_regression:
            raise ValueError(
                'Inconsistent is_regression={} passed. '
                'You don\'t have to pass it when using scikit-learn API'
                .format(is_regression))
        is_regression = _is_regression
    return booster, is_regression'''


def generator(ids, seqs,para=False):
    '''
    For given sequence, generate features
    '''
    val = {
        'mol_weight':[],
        'aaf':[],
        'pos_sub':[],
        'small_sub':[],
        'negative_sub':[],
        'hydrophobic_sub':[],
        'polar_sub':[],
        'pos_freq':[],
        'negative_freq':[],
        'hydrophobic_freq':[],
        'small_freq':[],
        'polar_freq':[],
        'small_c':[],
        'small_n':[],
        'positive_c':[],
        'positive_n':[],
        'negative_c':[],
        'negative_n':[],
        'polar_c':[],
        'polar_n':[],
        'hydrophobic_c':[],
        'hydrophobic_n':[],
        'W':[],
        'F':[],
        'R':[],
    }
    slc = {}

    dic = {
        'X': 'Q',
        'Z': 'E',
        'B': 'N',
        'J': 'L',
        'O': 'K'
    }

    def gen_seq_feature(val, seq):
        val['mol_weight'].append(gen_mol_weight(seq))
        val['aaf'].append(aafreq(seq))
        val['pos_sub'].append(sub_freq(seq, 'positive', 5))
        val['small_sub'].append(sub_freq(seq, 'small', 5))
        val['negative_sub'].append(sub_freq(seq, 'negative', 5))
        val['hydrophobic_sub'].append(sub_freq(seq, 'hydrophobic', 5))
        val['polar_sub'].append(sub_freq(seq, 'polar', 5))
        val['pos_freq'].append(chem_freq(seq, 'positive'))
        val['polar_freq'].append(chem_freq(seq, 'polar'))
        val['small_freq'].append(chem_freq(seq, 'small'))
        val['hydrophobic_freq'].append(chem_freq(seq, 'hydrophobic'))
        val['negative_freq'].append(chem_freq(seq, 'negative'))
        val['small_c'].append(gen_terminus_freq(seq, ['small'])[0])
        val['small_n'].append(gen_terminus_freq(seq, ['small'])[1])
        val['polar_c'].append(gen_terminus_freq(seq, ['polar'])[0])
        val['polar_n'].append(gen_terminus_freq(seq, ['polar'])[1])
        val['positive_c'].append(gen_terminus_freq(seq, ['positive'])[0])
        val['positive_n'].append(gen_terminus_freq(seq, ['positive'])[1])
        val['hydrophobic_c'].append(gen_terminus_freq(seq, ['hydrophobic'])[0])
        val['hydrophobic_n'].append(gen_terminus_freq(seq, ['hydrophobic'])[1])
        val['negative_c'].append(gen_terminus_freq(seq, ['negative'])[0])
        val['negative_n'].append(gen_terminus_freq(seq, ['negative'])[1])
        global count
        count+=1
        #print(count)
        #print(len(val['aaf']),len(val['aaf']),val['aaf'])
        #for datasets in SLC:
        #    slc[datasets] = np.array(val['aaf'])[:, SLC.index(datasets)]
        #val.update(slc)
        val['W'].append(aafreq(seq)[SLC.index('W')])
        val['F'].append(aafreq(seq)[SLC.index('F')])
        val['R'].append(aafreq(seq)[SLC.index('R')])
        return val

    #ids, seqs = read_fasta(inputs)
    for seq in seqs:
        if len(seq) < 20:
            raise ValueError("Cannot handle Sequence length shorter than 20")
        seq = string_replace(seq, dic)
        gen_seq_feature(val, seq)

    return val, ids

def read_fasta(inputs):
    res = re.findall(">.*?\n|(?:[^>].*?\n)+", inputs)
    ids = res[0::2]
    seqs = res[1::2]
    ids = [i.split(' ')[0].split('>')[1] if (' ' in i) == True else i[1:-1] for i in ids]

    seqs = [('').join(i.split('\n')) for i in seqs]
    n_seq = len(ids)
    if n_seq > 100: #the maximum number of query sequence is 100
        raise ValueError('The maximum number of sequence is 100!')
    return ids, seqs

def string_replace(text, dic):
    '''
    Replace uncommon amino acids
    '''
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def fill_aa_dict(in_dict):
    '''
    Some sequences might not have all the amino acids.
    The freq for missing ones are filled with zeros.
    '''
    SLC = 'ILVFMCAGPTSYWQNHEDKRU'
    if len(in_dict) < 21:
        exist_aa = in_dict.keys()
        for s in set(SLC)-set(exist_aa):
            in_dict[s] = 0

    if len(in_dict) < 21:
        raise ValueError('not all amino acids are covered')
    return in_dict

def aafreq(seq):
    '''
    Generates the amino acid frequencies for entire sequences
    '''
    SLC = 'ILVFMCAGPTSYWQNHEDKRU'
    freq_dict = dict(Counter(seq))
    if len(freq_dict) < 21:
        exist_aa = freq_dict.keys()
        for s in set(list(SLC))-set(exist_aa):
            freq_dict[s] = 0
    #freq_dict = fill_aa_dict(freq_dict)
    freq_list = [float(freq_dict[SLC[i]])/float(len(seq)) for i in range(21)]
    return freq_list

def sub_freq(seq, chem, num_slice):
    '''
    This function calculates the desired chemical frequencies for
    specified number of slices.

    Parameters:

        1. seq: 'str' the input sequence
        2. chem: 'str' desired chemical feature frequency
        3. num_slice: 'int' desired number of slices for each sequence
    '''
    SLC = list('ILVFMCAGPTSYWQNHEDKRU')
    chem_dict = {
        'small': 'TDNGASPC',       #small amino acids
        'hydrophobic': 'ACILVFWMP',  #reference: IARC TP53 database
        'polar': 'RKDENQ',    #clearly polar, reference: russelllab.org
        'positive': 'RHK',
        'negative': 'DE'
        }
    subseq_len = int(len(seq)/num_slice)
    subseq = [seq[i*subseq_len:(i+1)*subseq_len] for i in range(num_slice - 1)]
    subseq.append(seq[(num_slice - 1) * subseq_len:])

    feature_list = []
    for i in range(num_slice):
        freq_dict = dict(Counter(subseq[i]))
        freq_dict = fill_aa_dict(freq_dict)
        feature_list.append(sum([freq_dict[j] for j in chem_dict[chem]])/float(len(subseq[i])))
    return feature_list

def chem_freq(seq, chem):
    SLC = list('ILVFMCAGPTSYWQNHEDKRU')
    chem_dict = {
        'small': 'TDNGASPC',       #small amino acids
        'hydrophobic': 'ACILVFWMP',  #reference: IARC TP53 database
        'polar': 'RKDENQ',    #clearly polar, reference: russelllab.org
        'positive': 'RHK',
        'negative': 'DE'
        }

    freq_dict = dict(Counter(seq))
    freq_dict = fill_aa_dict(freq_dict)
    feature = sum([freq_dict[j] for j in chem_dict[chem]])/float(len(seq))
    return feature

def gen_mol_weight(seq):
    'Generate the molecular_weight for given sequence'
    mean_mammal, std_mammal = 55463.1246855, 56075.9002019
    res = ProtParam.ProteinAnalysis(seq)
    mw = res.molecular_weight()
    #print mw_list
    mw_feature = (mw - mean_mammal)/std_mammal
    return mw_feature

def gen_terminus_freq(seq, chems):
    SLC = list('ILVFMCAGPTSYWQNHEDKRU')
    chem_dict = {
        'small': 'TDNGASPC',       #small amino acids
        'hydrophobic': 'ACILVFWMP',  #reference: IARC TP53 database
        'polar': 'RKDENQ',    #clearly polar, reference: russelllab.org
        'positive': 'RHK',
        'negative': 'DE'
        }
    feat_nterm = []
    feat_cterm = []
    for chem in chems:
        if len(seq) < 50:
            nterm = seq
            cterm = seq
        else:
            nterm = seq[:50]
            cterm = seq[-50:]
        freq_nterm = fill_aa_dict(dict(Counter(nterm)))
        freq_cterm = fill_aa_dict(dict(Counter(cterm)))
        feat_nterm.append(sum([freq_nterm[j] for j in chem_dict[chem]])/float(len(nterm)))
        feat_cterm.append(sum([freq_cterm[j] for j in chem_dict[chem]])/float(len(cterm)))
    return (feat_cterm[0], feat_nterm[0])


#protein_df = pd.read_csv('uniprot2.tsv',sep='\t')
#ids_tmp = list(protein_df['Entry'])
#seqs_tmp = list(protein_df['Sequence'])
#
#ids=[]
#seqs=[]
#
#for i in range(len(ids_tmp)):
#    if len(seqs_tmp[i])>=20:
#        ids.append(ids_tmp[i])
#        seqs.append(seqs_tmp[i])
#
#f,l = generator(ids,seqs)
##print(f)
#print(f['MKHNMPLTEEEERKMYNNVLKTGWTLDQVKERLEKGDETLGDHEDRLVSLEGDQKLLSGKMGMIVLGMSVCFTAALHAVGWILSHFCKS'])
