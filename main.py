# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:19:37 2018

@author: user
"""
import numpy as np
import string
import collections
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import random

def split_data(sentset):
    samplist = random.sample(range(0,len(t_all)),9000)
    #print(samplist)
    test_data = []
    train_data = [] 
    for idx in samplist:
        test_data.append(t_all[idx])
    for idx in range(0, len(t_all)): 
        if idx not in samplist:
            train_data.append(t_all[idx])
    return np.mat(train_data), np.mat(test_data)

def sentences(string_list, word2idx, label): 
    sents = []
    for line in string_list:
        line = line.split()
        line = filter_text(line)
        line = filter_text_by_vocab(line)
  
        zovl = [0 for _ in range(len(word2idx))]    
        for word in line:
            indx = word2idx[word]
            zovl[indx] = 1
            
        zovl.append(label)
        sents.append(zovl)
  
    return np.asarray(sents)

def filter_text(texts):
    texts = [word.lower() for word in texts]
    puns = [pun for pun in string.punctuation if pun !='?']
    table = str.maketrans('', '', str(puns))
    list_texts = [w.translate(table) for w in str(texts).split()]
    stopwrds = ['','the','a','an','ufeffthe','to','of','and','or','is','in','that','its',
            'it','as','with','this','for','be','on','by','about','at','but','film', 
            'movie','you','his','has','have','into','films','are','from','not',
            'more','one','some','their','than','all','so','who','what','way',
            'movies','been','your','there','us','no','even','will','story',
            'make','when','makes','he','they','her','them','thats','those',
            'isnt','these','me','before','where','whose','him','arent',
            'whats','men','john','each','viewers','mostly','put','sort',
            'version','next','level','youve','face','along','whether',
            'looks','bring','steven','called','substance','shes','manner'
            'parts','person','couldnt','jackson','viewing','places','holds',
            'said','saw','extremely','word','runs','tells','sets','flicks',
            'become','looking','didnt']
      
    list_texts = [word for word in list_texts if word not in stopwrds]
    return list_texts

def filter_text_by_vocab(texts):
    texts = [word for word in texts if word in vocab]
    return texts
'''
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    parameters = [
      {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf']
      },
      {
        'C': [1, 5, 9, 13, 17, 19],
        'kernel': ['linear']
      }
    ]
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=8)
    clf.fit(train_x, train_y)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    return best_model
    
    def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(C=19, kernel='sigmoid', gamma='auto', coef0=1, probability=False)
    model.fit(train_x, train_y)
    return model
'''

def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    parameters = {'C': range(1, 13, 19)}
 
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=4)
    clf.fit(train_x, train_y)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    return best_model


def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=1200, max_depth=11, learning_rate=0.08, 
                                       subsample=1, min_samples_split=850, max_leaf_nodes=50) 
    parameters = {'max_depth': range(3,7,11)}
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=4)
    clf.fit(train_x, train_y)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    return best_model
#n_estimators=110, max_depth=11, learning_rate=0.8
#n_estimators=120, max_depth=11, learning_rate=0.7
#n_estimators=125, max_depth=11, learning_rate=0.65
#n_estimators=127, max_depth=11, learning_rate=0.55, min_samples_split=800
#n_estimators=130, max_depth=11, learning_rate=0.5, min_samples_split=900

def train_predict(times,t_all):
    avg_accuracy = 0
    for i in range(times):
        train_data, test_data = split_data(t_all)
        #print(train_data[:,:-1])
        #print
        #print(test_data.shape)
        model_svm = svm_classifier(train_data[:,:-1], train_data[:,-1])
        model_gbdt= gradient_boosting_classifier(train_data[:,:-1], train_data[:,-1])
        print('fitted')
        predict_svm = model_svm.predict(test_data[:,:-1])
        #predict_gbdt = model_gbdt.predict(test_data[:,:-1])
        #precision = metrics.precision_score(test_data[:,-1], predict)
        #recall = metrics.recall_score(test_data[:,-1], predict)
        #print ('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy_svm = metrics.accuracy_score(test_data[:,-1], predict_svm)
        #accuracy_gbdt = metrics.accuracy_score(test_data[:,-1], predict_gbdt)
        print ('accuracy_svm: %.2f%%' % (100 * accuracy_svm))
        print
        #print ('accuracy_gbdt: %.2f%%' % (100 * accuracy_gbdt))   
        avg_accuracy+=accuracy_gbdt
    avg_accuracy = avg_accuracy/times
    print ('avg_accuracy_gbdt: %.2f%%' % (100 * avg_accuracy)) 
    return accuracy_gbdt

def gs_train_predict(t_all):
    train_data, test_data = split_data(t_all)
    model_svm = svm_classifier(train_data[:,:-1], train_data[:,-1])
    print('fitted')
    predict_svm = model_svm.predict(test_data[:,:-1])
    accuracy_svm = metrics.accuracy_score(test_data[:,-1], predict_svm)
    print ('accuracy_svm: %.2f%%' % (100 * accuracy_svm))
    return accuracy_svm

def cvSVM_train_predict(t_all):
    print("SVM Cross Validation")
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', C=15)
    scores = cross_val_score(clf, t_all[:,:-1], t_all[:,-1], cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return
#C=1 kernel='linear'

def cvGBDT_train_predict(t_all):
    print("GBDT Cross Validation")
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=1, 
                                       subsample=1, min_samples_split=300) #, min_samples_split=None
    scores = cross_val_score(clf, t_all[:,:-1], t_all[:,-1], cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return
#n_estimators=100, max_depth=7, learning_rate=0.9, subsample=0.9  =>Accuracy: 0.70 (+/- 0.00)
#n_estimators=100, max_depth=7, learning_rate=1, subsample=1  =>Accuracy: 0.70 (+/- 0.01)                                       
#n_estimators=100, max_depth=7, learning_rate=1, subsample=1, min_samples_split=100  =>Accuracy: 0.70 (+/- 0.01)
#n_estimators=100, max_depth=7, learning_rate=1, subsample=1, min_samples_split=300  =>Accuracy: 0.71 (+/- 0.01)


with open('data/rt-polarity.pos.txt', encoding='utf-8') as f:
        neg_texts = f.read().splitlines()   # f.read()读取整个文件, splitlines() 按行分割，形成list
with open('data/rt-polarity.neg.txt', encoding='utf-8') as f:
        pos_texts = f.read().splitlines()

list_texts = filter_text(neg_texts+pos_texts)
counter = collections.Counter(list_texts)
list_word_count = counter.most_common(1000)
#这个列表要转成词为键、词排序为值的字典
vocab = [x[0] for x in list_word_count]
#print(vocab[310:])
word2idx = {c: i for i, c in enumerate(vocab)}
#print(word2idx['film'])

t_pos = sentences(pos_texts, word2idx, 1)
t_neg = sentences(neg_texts, word2idx, 0)
t_all = np.row_stack((t_pos,t_neg))
#the last column is set to be label !


 
#train_predict(5,t_all)
#gs_train_predict(t_all)
#cvSVM_train_predict(t_all)
cvGBDT_train_predict(t_all)

'''
texts = [word.lower() for word in neg_texts+pos_texts]

puns = [pun for pun in string.punctuation if pun !='?']
table = str.maketrans('', '', str(puns))
list_texts = [w.translate(table) for w in str(texts).split()]
stopwrds = ['','the','a','an','ufeffthe','to','of','and','or','is','in','that','its',
            'it','as','with','this','for','be','on','by','about','at']
list_texts = [word for word in list_texts if word not in stopwrds]

stopwrds = ['','the','a','an','ufeffthe','to','of','and','or','is','in','that','its',
            'it','as','with','this','for','be','on','by','about','at']
word2idx = {}
    for line in see:
        for word in line.split():
            if not word in word2idx:
                word2idx[word] = len(word2idx) + 1
print()

enc = preprocessing.OneHotEncoder()
enc.fit([t_pos[0],t_pos[1]]) 
see = enc.transform([t_pos[0]]).toarray()
'''


