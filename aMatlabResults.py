import pandas as pd
import numpy as np

import sklearn.metrics
file='zzmatlabcomparsionresult.xlsx'
data = pd.read_excel(file)
data=np.array(data.values)
y_real=data[:,0]
y_st_svm=data[:,1]
y_wt_rbf=data[:,2]
y_real = y_real.astype(np.int64)
y_st_svm = y_st_svm.astype(np.int64)
y_wt_rbf = y_wt_rbf.astype(np.int64)

# y_real=np.array(y_real)
# y_st_svm=np.array(y_st_svm)
# y_wt_rbf=np.array(y_wt_rbf)
'''
WT+RBF
'''
correct = (y_real == y_wt_rbf).sum()
total = len(y_real)
Accuracy =(correct / total)


micro_precision = sklearn.metrics.precision_score(y_real, y_wt_rbf, labels=None, average='micro',
                                sample_weight=None)
macro_precision = sklearn.metrics.precision_score(y_real, y_wt_rbf, labels=None, average='macro',
                                sample_weight=None)
micro_f1 = sklearn.metrics.f1_score(y_real, y_wt_rbf, labels=None, average='micro',
                                sample_weight=None)
macro_f1 = sklearn.metrics.f1_score(y_real, y_wt_rbf, labels=None, average='macro',
                                sample_weight=None)
micro_recall = sklearn.metrics.recall_score(y_real, y_wt_rbf, labels=None, average='micro',
                                sample_weight=None)
macro_recall = sklearn.metrics.recall_score(y_real, y_wt_rbf, labels=None, average='macro',
                                sample_weight=None)
print("WT+RBF")
print("test Accuraccy",Accuracy)
print("test micro_precision", micro_precision)
print("test macro_precision", macro_precision)
print("test micro_recall", micro_recall)
print("test macro_recall", macro_recall)
print("test micro_f1", micro_f1)
print("test macro_f1", macro_f1)

'''
ST+SVM
'''
correct = (y_real == y_st_svm).sum()
total = len(y_real)
Accuracy =(correct / total)


micro_precision = sklearn.metrics.precision_score(y_real, y_st_svm, labels=None, average='micro',
                                sample_weight=None)
macro_precision = sklearn.metrics.precision_score(y_real, y_st_svm, labels=None, average='macro',
                                sample_weight=None)
micro_f1 = sklearn.metrics.f1_score(y_real, y_st_svm, labels=None, average='micro',
                                sample_weight=None)
macro_f1 = sklearn.metrics.f1_score(y_real, y_st_svm, labels=None, average='macro',
                                sample_weight=None)
micro_recall = sklearn.metrics.recall_score(y_real, y_st_svm, labels=None, average='micro',
                                sample_weight=None)
macro_recall = sklearn.metrics.recall_score(y_real, y_st_svm, labels=None, average='macro',
                                sample_weight=None)
print("ST+SVM")
print("test Accuraccy",Accuracy)
print("test micro_precision", micro_precision)
print("test macro_precision", macro_precision)
print("test micro_recall", micro_recall)
print("test macro_recall", macro_recall)
print("test micro_f1", micro_f1)
print("test macro_f1", macro_f1)


# WT+RBF
# test Accuraccy 0.59375
# test micro_precision 0.59375
# test macro_precision 0.7765151515151515
# test micro_recall 0.59375
# test macro_recall 0.5818181818181819
# test micro_f1 0.59375
# test macro_f1 0.5587453482190324
# ST+SVM
# test Accuraccy 0.6875
# test micro_precision 0.6875
# test macro_precision 0.8333333333333334
# test micro_recall 0.6875
# test macro_recall 0.6969696969696969
# test micro_f1 0.6875
# test macro_f1 0.6422466422466422
