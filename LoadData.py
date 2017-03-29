import sys,os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
sys.path.append('ClonedModel/wmModel/wiki-detox/src/modeling/')


# Some Helper Functions with WikiMedia


def empirical_dist(l, w = 0.0, index = None):
    """
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
    """
    Take the most common label from all labels with the same rev_id.
    
    Return:
    =======
    s = an array of integers of 0 or 1
    """
    s = l.groupby(l.index).apply(lambda x:x.value_counts().index[0])
    s.name = 'y'
    return s



def one_hot(y):
    """
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
    y_oh[list(range(m)), idxs] = 1
    return y_oh



def load_and_parse_training(data_dir, task, data_type):
    ''' Load and Parse training Data'''
    
    # comments is X, annotations (labels) are y
    COMMENTS_FILE = "%s_annotated_comments.tsv" % task
    LABELS_FILE = "%s_annotations.tsv" % task
    
    comments = pd.read_csv(os.path.join(data_dir, COMMENTS_FILE), sep = '\t', index_col = 0)
    annotations = pd.read_csv(os.path.join(data_dir, LABELS_FILE),  sep = '\t', index_col = 0)
    
    # remove special newline and tab tokens
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    
    # sort X
    X = comments.sort_index()['comment'].values
    
    if(data_type == 'empirical'):
        labels = empirical_dist(annotations[task])
        y = labels.sort_index().values        
    elif(data_type == 'onehot'):
        y = plurality(annotations[task])
        
        
    assert(X.shape[0] == y.shape[0])    
    return X, y  
    


    
def load_Onehot_train_test_split(task):
    '''
    Load Data using Load_and_Parse_Training
        in empirical form
        and made into onehot
        (so that easy to change to emp again)
    Train_Split using sklearn
    
    Return:
    =======
    DATA: dictionary{'Training', 'Testing'}
    
    '''
    DATA_DIR = 'TalkData/computed_dataset/'
    [X,yEmp] = load_and_parse_training(DATA_DIR, task, 'empirical')
    yOneHot = one_hot(yEmp)
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, yOneHot, 
                                        test_size=0.15, 
                                        random_state=0
                                        )


    DATA_TRAINING = pd.DataFrame([X_train,y_train]).T
    DATA_TESTING = pd.DataFrame([X_test,y_test]).T
    
    DATA_TRAINING.columns = ['Text','Category']
    DATA_TRAINING['Category'] = DATA_TRAINING['Category'].apply(
                lambda y: 'notAttack' if y.argmax() == 0 else 'Attack')

    DATA_TESTING.columns = ['Text','Category']
    DATA_TESTING['Category'] = DATA_TESTING['Category'].apply(
                lambda y: 'notAttack' if y.argmax() == 0 else 'Attack')
    
    return {
            'Training': DATA_TRAINING,
            'Testing': DATA_TESTING
        }
