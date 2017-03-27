from __future__ import absolute_import
import sys,os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(u'ClonedModel/wmModel/wiki-detox/src/modeling/')


# Some Helper Functions with WikiMedia


def empirical_dist(l, w = 0.0, index = None):
    u"""
    Compute empirical distribution over all classes
    using all labels with the same rev_id
    """
    if not index:
        index = sorted(list(set(l.dropna().values)))

    data = {}
    for k, g in l.groupby(l.index):
        data[k] = g.value_counts().reindex(index).fillna(0) + w

    labels = pd.DataFrame(data).T
    labels = labels.fillna(0)
    labels = labels.div(labels.sum(axis=1), axis=0)
    return labels



def plurality(l):
    u"""
    Take the most common label from all labels with the same rev_id.
    
    Return:
    =======
    s = an array of integers of 0 or 1
    """
    s = l.groupby(l.index).apply(lambda x:x.value_counts().index[0])
    s.name = u'y'
    return s



def one_hot(y):
    u"""
    Return:
    =======
    y_oh = an array of vectors (one-hot vectors)
    """
    m = y.shape[0]
    
    if len(y.shape) == 1:
        n = len(set(y.ravel()))
        idxs = y.astype(int)
    else:
        idxs = y.argmax(axis = 1)
        n = y.shape[1]

    y_oh = np.zeros((m, n))
    y_oh[range(m), idxs] = 1
    return y_oh



def load_and_parse_training(data_dir, task, data_type):
    u''' Load and Parse training Data'''
    
    # comments is X, annotations (labels) are y
    COMMENTS_FILE = u"%s_annotated_comments.tsv" % task
    LABELS_FILE = u"%s_annotations.tsv" % task
    
    comments = pd.read_csv(os.path.join(data_dir, COMMENTS_FILE), sep = u'\t', index_col = 0)
    annotations = pd.read_csv(os.path.join(data_dir, LABELS_FILE),  sep = u'\t', index_col = 0)
    
    # remove special newline and tab tokens
    comments[u'comment'] = comments[u'comment'].apply(lambda x: x.replace(u"NEWLINE_TOKEN", u" "))
    comments[u'comment'] = comments[u'comment'].apply(lambda x: x.replace(u"TAB_TOKEN", u" "))
    
    # sort X
    X = comments.sort_index()[u'comment'].values
    
    if(data_type == u'empirical'):
        labels = empirical_dist(annotations[task])
        y = labels.sort_index().values        
    elif(data_type == u'onehot'):
        y = plurality(annotations[task])
        
        
    assert(X.shape[0] == y.shape[0])    
    return X, y  
    


    
def load_Onehot_train_test_split(task):
    u'''
    Load Data using Load_and_Parse_Training
        in empirical form
        and made into onehot
        (so that easy to change to emp again)
    Train_Split using sklearn
    
    Return:
    =======
    DATA: dictionary{'Training', 'Testing'}
    
    '''
    DATA_DIR = u'TalkData/computed_dataset/'
    [X,yEmp] = load_and_parse_training(DATA_DIR, task, u'empirical')
    yOneHot = one_hot(yEmp)
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, yOneHot, 
                                        test_size=0.15, 
                                        random_state=0
                                        )


    DATA_TRAINING = pd.DataFrame([X_train,y_train]).T
    DATA_TESTING = pd.DataFrame([X_test,y_test]).T
    
    DATA_TRAINING.columns = [u'Text',u'Category']
    DATA_TRAINING[u'Category'] = DATA_TRAINING[u'Category'].apply(
                lambda y: u'notAttack' if y.argmax() == 0 else u'Attack')

    DATA_TESTING.columns = [u'Text',u'Category']
    DATA_TESTING[u'Category'] = DATA_TESTING[u'Category'].apply(
                lambda y: u'notAttack' if y.argmax() == 0 else u'Attack')
    
    return {
            u'Training': DATA_TRAINING,
            u'Testing': DATA_TESTING
        }
