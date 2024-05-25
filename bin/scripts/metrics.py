import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy(tp,tn,fp,fn):
    if (tp+tn+fp+fn)==0:
        return 0
    return (tp+tn)/(tp+tn+fp+fn)

def specifity(tp,tn,fp,fn):
    if (tn+fp)==0:
        return 0
    return tn/(tn+fp)

def sensitivity(tp,tn,fp,fn):
    if tp+fn==0:
        return 0
    return tp/(tp+fn)

def fdr(tp,tn,fp,fn):
    if tp+fp==0:
        return 1
    return  fp/(tp+fp)

def frr(tp,tn,fp,fn):
    if tn+fn==0:
        return 1
    return fn/(fn+tn)

def f1(tp,tn,fp,fn):
    if (2*tp+fp+fn)==0:
        return 0
    return 2*tp/(2*tp+fp+fn)

def mcc(tp,tn,fp,fn):
    if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)<=0:
        return -1
    return (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
def get_tp_tn_fp_fn(predictions, ground_truth):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    false_positives = ((predictions == 1) & (ground_truth == 0)).sum().item()

    true_positives = ((predictions == 1) & (ground_truth == 1)).sum().item()

    false_negatives =  ((predictions == 0) & (ground_truth == 1)).sum().item()

    true_negatives =  ((predictions == 0) & (ground_truth == 0)).sum().item()

    return true_positives,true_negatives,false_positives,false_negatives

def calc_metrics(tp,tn,fp,fn):


    metrics = {}

    metrics['accuracy']= round(accuracy(tp,tn,fp,fn),3)
    metrics['sensitivity']= round(sensitivity(tp,tn,fp,fn),3)
    metrics['specificity']= round(specifity(tp,tn,fp,fn),3)
    metrics['fdr']=round(fdr(tp, tn, fp, fn),3)
    metrics['frr']=round(frr(tp, tn, fp, fn),3)
    metrics['f1']=round(f1(tp, tn, fp, fn),3)
    metrics['mcc']=round(mcc(tp, tn, fp, fn),3)
    metrics['numbers']= np.array((tp,tn,fp,fn))

    return metrics


def metrics(predictions, ground_truth):


    tp,tn,fp,fn = get_tp_tn_fp_fn(predictions,ground_truth)

    metrics = {}

    metrics['accuracy']= accuracy(tp,tn,fp,fn)
    metrics['specificity']= specifity(tp,tn,fp,fn)
    metrics['sensitivity']= sensitivity(tp,tn,fp,fn)
    metrics['fdr']=fdr(tp, tn, fp, fn)
    metrics['frr']=frr(tp, tn, fp, fn)
    metrics['f1']=f1(tp, tn, fp, fn)
    metrics['mcc']=mcc(tp, tn, fp, fn)

    return metrics
