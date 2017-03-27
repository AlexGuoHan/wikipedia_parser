from __future__ import with_statement
from __future__ import absolute_import
import sys, os
import joblib
import pandas as pd

import time
from bs4 import BeautifulSoup
from argparse import ArgumentParser
from difflib import SequenceMatcher
from sklearn.pipeline import Pipeline
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.wrappers.scikit_learn import KerasClassifier


import LoadData
import CleanTextData
from io import open


class Stopwatch(object):
    start_time=None
    def go(self,msg=u''):
        if msg:
            print msg,
        self.start_time=time.time()
        sys.stdout.flush()
    def stop(self,msg=u''):
        if msg:
            print u"{}: {} seconds".format(msg,time.time()-self.start_time)
        else:
            print u"Elapsed time: {} seconds".format(time.time()-self.start_time)
        sys.stdout.flush()
    def check(self):
        return time.time()-self.start_time
tic=Stopwatch()


def argParser():
    parser = ArgumentParser()
    parser.add_argument(u'--wikiModelDir', type=unicode,
                        dest=u'wikiModelDir', 
                        help=u'directory to wikiMedia models',
                        required=True)
    parser.add_argument(u'--trainDataDir', type=unicode, 
                        dest=u'trainDataDir', 
                        help=u'directory to training data', 
                        required=True)
    parser.add_argument(u'--dataFileList', type=unicode, 
                        dest=u'dataFileList', 
                        help=u'directory to data file list', 
                        required=True)
    parser.add_argument(u'--dataFileDir', type=unicode, 
                        dest=u'dataFileDir', 
                        help=u'directory to the data',
                        default=None)
    parser.add_argument(u'--cpu', type=int,
                        dest=u'cpu',
                        help=u'number of cpu to deploy, 0 for max',
                        required=True)
    return parser;



def main():
    parser = argParser()
    args = parser.parse_args()
    
    wikiModelDir = args.wikiModelDir
    trainDataDir = args.trainDataDir
    dataFileList = args.dataFileList
    dataFileDir = args.dataFileDir
    num_cpus = args.cpu
    
    # load modules
    load_modules(wikiModelDir)
    
    tic.go(u'LOADING MODELS')
    
    # load losgistic model
    global logisticModel
    logisticModel = load_logistic_char_model(wikiModelDir)
    
    # load and train mlp model
    global mlpModel 
    mlpModel = load_mlp_char_model(wikiModelDir, trainDataDir)
    
    tic.stop()
    
    dataFiles = []
    # parallelize over multiple cpu
    with open(dataFileList,u'r') as f:
            for line in f:
                if(dataFileDir != None):
                    dfile = os.path.join(dataFileDir, line.strip(u'\n'))
                    dataFiles.append(dfile)
                else:
                    dfile = line.strip(u'\n')
                    dataFiles.append(dfile)
                assert os.path.exists(dfile), u'File Not Exist'
    assert cpu_count() >= num_cpus,u'more cpu than available'
    if(num_cpus <= 0): num_cpus = cpu_counts()
    print u'CPU: %d' % num_cpus
    
    pool = Pool(num_cpus)
    pool.map(load_apply_save,dataFiles)
    pool.join()
    pool.close()


    

def load_apply_save(dataDir):
    
    # load data
    tic.go(u'LOADING & CLEANING DATA %s'%(dataDir))
    raw_data = pd.read_csv(dataDir, sep=u'\t')
    cleaned_data = clean_and_diff(raw_data)
    tic.stop()
    
    # apply two models
    tic.go(u'APPLYING Logistic MODELS')
    cleaned_data = apply_models_DF(cleaned_data, logisticModel)
    tic.stop()
    tic.go(u'APPLYING MLP MODELS')
    cleaned_data = apply_models_DF(cleaned_data, mlpModel)
    tic.stop()
    
    # save
    filename = os.path.basename(dataDir)
    data.to_csv(u'predicted_%s'%filename)
    
    
    
    

def load_modules(wikiModelDir):
    u''' This function will import modules based on wmModeiDir variable'''
    assert os.path.exists(wikiModelDir), u'wikiModelDir Not Exist'
    
    global ngram
    global load_comments_and_labels, assemble_data, one_hot
    global make_mlp, DenseTransformer
    global save_pipeline, load_pipeline
    global diff_utils
    
    sys.path.append(os.path.join(wikiModelDir,u'wiki-detox/src/modeling'))
    sys.path.append(os.path.join(wikiModelDir,u'wiki-detox/src/data_generation'))
    import ngram
    from baselines import load_comments_and_labels, assemble_data, one_hot
    from deep_learning import make_mlp, DenseTransformer
    from serialization import save_pipeline, load_pipeline
    import diff_utils
    
    

def load_logistic_char_model(wikiModelDir):
    u''' Load and return the pretrained logistic character module '''
    
    # load pretrained model
    attackModelDir = os.path.join(wikiModelDir,
        u'wiki-detox/app/models/attack_linear_char_oh_pipeline.pkl')
    aggrModelDir = os.path.join(wikiModelDir,
        u'wiki-detox/app/models/aggression_linear_char_oh_pipeline.pkl')
    
    assert os.path.isfile(attackModelDir), u'Attack Model NOT found'
    assert os.path.isfile(aggrModelDir), u'Aggression Model NOT found'
    
    return {
        u'attackModel': joblib.load(attackModelDir),
        u'aggrModel': joblib.load(aggrModelDir)
    }
    
    
def load_mlp_char_model(wikiModelDir, trainDataDir):
    
    # load best hyper-parameters
    cvResultsDir = os.path.join(wikiModelDir, 
                     u'wiki-detox/src/modeling/cv_results.csv')
    
    bestParams = load_best_params(cvResultsDir,u'mlp',u'char',u'ed')
    PIPELINE = Pipeline([
                        (u'vect', CountVectorizer()),
                        (u'tfidf', TfidfTransformer()),
                        (u'to_dense', DenseTransformer()), 
                        (u'clf', KerasClassifier(build_fn=make_mlp, 
                                                output_dim = 2, 
                                                verbose=False))]) 
    PIPELINE.set_params(**bestParams)
    
    # train models
    trainData = load_training_data(trainDataDir)
    
    attackModel = PIPELINE
    aggrModel = PIPELINE
    
    attackModel.fit(trainData[u'attackTrainData'][u'X'],
                    trainData[u'attackTrainData'][u'y'])
    aggrModel.fit(trainData[u'aggrTrainData'][u'X'],
                    trainData[u'aggrTrainData'][u'y'])

    return {
        u'attackModel': attackModel,
        u'aggrModel': aggrModel
    }


def load_best_params(cv_results_dir, model_type, ngram_type, label_type):
    u'''
    Input:
    ======
    cv_result_dir: the directory to "cv_result" file of WikiMedia model
    '''
                               
    import json
    
    cv_results = pd.read_csv(cv_results_dir)
    query = u"model_type == \'%s\' and ngram_type == \'%s\' and label_type == \'%s\'" % (
                                    model_type, ngram_type, label_type)
        
    params = cv_results.query(query)
    params = params.loc[:,u'best_params'].iloc[0]
    return json.loads(params)


def load_training_data(trainDataDir):
    assert os.path.exists(trainDataDir), u'trainDataDir Not Exist'
    attackTrainData = LoadData.load_and_parse_training(trainDataDir,
                                                       u'attack',
                                                       u'empirical')
    aggrTrainData = LoadData.load_and_parse_training(trainDataDir,
                                                     u'aggression',
                                                     u'empirical')
    return {
        u'attackTrainData': {
                              u'X': attackTrainData[0],
                              u'y': attackTrainData[1]
                            },
        u'aggrTrainData':   {
                              u'X': aggrTrainData[0],
                              u'y': aggrTrainData[1]
                            }
    }
                               

def get_diff(old, new, char_threshold = 5, ratio_threshold = 0.5):
    u''' find diff using exhaustive search, not recommemded '''
    # find the lines with length > threshold characters
    old_lines = [o for o in old.splitlines() if len(o) > char_threshold] 
    new_lines = [n for n in new.splitlines() if len(n) > char_threshold]
   
    diff = []    
    for new_line in new_lines:
        will_append = True
        for old_line in old_lines:
            append = SequenceMatcher(None, new_line, old_line).ratio() < ratio_threshold
            will_append = min(will_append,append)
        if(will_append is True): diff.append(new_line)
    return u'\n'.join(diff)


def clean_and_diff(data, method=u'quick_2', verbose=False):
    u''' taking the diff and clean the text column
    
    Return:
    =======
    data: a DataFrame with the cleaned text on 'clean_text' column
    
    '''
    
    # Clean the data
    assert u'title' in data.columns.tolist(), u'DataFrame format Incorrect'
    assert u'text' in data.columns.tolist(), u'DataFrame format Incorrect'
    
    # use wikipedia's clean text data function
    data = CleanTextData.clean_and_filter(data, text_col=u'text', min_words=0,  min_chars=0)
    
    # their function will produce some columns we dont need
    data[u'clean_text'] = data[u'clean_diff']
    data = data.drop([u'diff',u'clean_diff'],1)
    
    assert u'diff' not in data.columns.tolist()
    assert u'clean_diff' not in data.columns.tolist()
    

    
    
    # Diff the data
    titles = data.title.unique()
    idx = 0
    # taking the diff for each title
    for title in titles:
        data_subset = data[data.title == title]
        text_diff = [data_subset.clean_text.iloc[0]]
        
        for idx in xrange(idx, idx + data_subset.shape[0] - 1 ):
            
            try:    
                new = data_subset.clean_text[idx + 1]
            except KeyError:
                if(verbose == True):
                    print u"text has deleted, changed to empty"
                new = u''
            
            try:
                old = data_subset.clean_text[idx]
            except KeyError:
                if(verbose == True):
                    print u"text has deleted, changed to empty"
                old = u''
                
                
            try:
                    delta_bytes = data_subset.byte[1 + idx]
            except KeyError:
                if(verbose == True):
                    print u"text has deleted, changed byte to 0"
                delta_bytes = 0
                
    
            if(type(new) is not unicode):
                if(verbose == True):
                    print u"text is not str: %s, changed to empty"%(new)
                new = u''
            if(type(old) is not unicode):
                if(verbose == True):
                    print u"text is not str: %s, changed to empty"%(old)
                old = u''
            
            # slow has better performance
            # quick works okay, but definitely need improvement
            if(method == u'slow'): 
                text_diff.append(get_diff(old,new))
            if(method == u'quick_1'): 
                text_diff.append(new.replace(old,u' ',1))
            if(method == u'quick_2'): 
                text_diff.append(new[len(old):])

        # data_subset.shape[0] - 1 + 1
        idx = idx + 2;
        data.loc[data.title == title,u'diff_text'] = pd.Series(text_diff)
    
    return data
                               
    
def apply_models_DF(df, model_dict, col=u'clean_text'):
    u''' Predict the probability of input data to be labelled
        'aggressive' or 'attack' using 
        
        Return:
        =======
        a data frame with pred scores attached
        
    '''
    
    texts = df[col]
    for task,model in model_dict.items():
        scores = model.predict_proba(texts)[:,1]
        df[u'%s_logistic_score'%(task)] = scores
    return df

def apply_models_text(text, model_dict):
    u''' Predict the probability of input texts to be labelled
        'aggressive' or 'attack'    
        
        Used for sanity check
    '''

    for task,model in model_dict.items():
        scores = model.predict_proba([text])[:,1]
        print u'%s_mlp_score: %f'%(task,scores)
    
    
    
    
    
    
if __name__ == u'__main__':
    main()
