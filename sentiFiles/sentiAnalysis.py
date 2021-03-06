import sys, os
import joblib
import pandas as pd

import time
import difflib
from bs4 import BeautifulSoup
from argparse import ArgumentParser
from sklearn.pipeline import Pipeline
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.wrappers.scikit_learn import KerasClassifier


import LoadData
import CleanTextData


class Stopwatch:
    start_time=None
    def go(self,msg=''):
        if msg:
            print(msg),
        self.start_time=time.time()
        sys.stdout.flush()
    def stop(self,msg=''):
        if msg:
            print("{}: {} seconds".format(msg,time.time()-self.start_time))
        else:
            print("Elapsed time: {} seconds".format(time.time()-self.start_time))
        sys.stdout.flush()
    def check(self):
        return time.time()-self.start_time
tic=Stopwatch()


def argParser():
    parser = ArgumentParser()
    parser.add_argument('--wikiModelDir', type=str,
                        dest='wikiModelDir', 
                        help='directory to wikiMedia models',
                        required=True)
    parser.add_argument('--trainDataDir', type=str, 
                        dest='trainDataDir', 
                        help='directory to training data', 
                        required=True)
    parser.add_argument('--dataFileList', type=str, 
                        dest='dataFileList', 
                        help='directory to data file list', 
                        required=True)
    parser.add_argument('--dataFileDir', type=str, 
                        dest='dataFileDir', 
                        help='directory to the data',
                        default=None)
    parser.add_argument('--cpu', type=int,
                        dest='cpu',
                        help='number of cpu to deploy, 0 for max',
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
    
    tic.go('LOADING MODELS')
    
    # load losgistic model
    global logisticModel
    logisticModel = load_logistic_char_model(wikiModelDir)
    
    # load and train mlp model
    global mlpModel 
    mlpModel = load_mlp_char_model(wikiModelDir, trainDataDir)
    
    tic.stop()
    
    dataFiles = []
    # parallelize over multiple cpu
    with open(dataFileList,'r') as f:
            for line in f:
                if(dataFileDir != None):
                    dfile = os.path.join(dataFileDir, line.strip('\n'))
                    dataFiles.append(dfile)
                else:
                    dfile = line.strip('\n')
                    dataFiles.append(dfile)
                assert os.path.exists(dfile), 'File Not Exist'
    assert cpu_count() >= num_cpus,'more cpu than available'
    if(num_cpus <= 0): num_cpus = cpu_counts()
    print('CPU: %d' % num_cpus)
    
    pool = Pool(num_cpus)
    pool.map(load_apply_save,dataFiles)
    # pool.join()
    # pool.close()


    

def load_apply_save(dataDir):
    
    # load data
    tic.go('LOADING & CLEANING DATA %s'%(dataDir))
    raw_data = pd.read_csv(dataDir, sep='\t')
    cleaned_data = diff_clean_text(raw_data, verbose=True)
    tic.stop()
    
    # if(status == 'non Empty'):
    
    assert 'Added' in cleaned_data.columns(), 'Data Format Error, no Added'
    assert 'Deleted' in cleaned_data.columns(), 'Data Format Error, no Deleted'
        
    # apply two models
    tic.go('APPLYING Logistic MODELS')
    cleaned_data = apply_models_DF(cleaned_data, 
                                   'logistic',
                                   logisticModel, 
                                   ['Added', 'Deleted'])
    
    tic.stop()
    
    tic.go('APPLYING MLP MODELS')
    cleaned_data = apply_models_DF(cleaned_data, 
                                    'mlp', 
                                    mlpModel, 
                                    ['Added', 'Deleted'])
    tic.stop()

    # save
    filename = os.path.basename(dataDir)
    cleaned_data.to_csv('predicted_%s'%filename, sep='\t')

    
    

def load_modules(wikiModelDir):
    ''' This function will import modules based on wmModeiDir variable'''
    assert os.path.exists(wikiModelDir), 'wikiModelDir Not Exist'
    
    global ngram
    global load_comments_and_labels, assemble_data, one_hot
    global make_mlp, DenseTransformer
    global save_pipeline, load_pipeline
    
    sys.path.append(os.path.join(wikiModelDir,u'wiki-detox/src/modeling'))
    sys.path.append(os.path.join(wikiModelDir,u'wiki-detox/src/data_generation'))
    import ngram
    from baselines import load_comments_and_labels, assemble_data, one_hot
    from deep_learning import make_mlp, DenseTransformer
    from serialization import save_pipeline, load_pipeline
    
    

def load_logistic_char_model(wikiModelDir):
    ''' Load and return the pretrained logistic character module '''
    
    # load pretrained model
    attackModelDir = os.path.join(wikiModelDir,
        'wiki-detox/app/models/attack_linear_char_oh_pipeline.pkl')
    aggrModelDir = os.path.join(wikiModelDir,
        'wiki-detox/app/models/aggression_linear_char_oh_pipeline.pkl')
    
    assert os.path.isfile(attackModelDir), 'Attack Model NOT found'
    assert os.path.isfile(aggrModelDir), 'Aggression Model NOT found'
    
    return {
        'attackModel': joblib.load(attackModelDir),
        'aggrModel': joblib.load(aggrModelDir)
    }
    
    
def load_mlp_char_model(wikiModelDir, trainDataDir):
    
    # load best hyper-parameters
    cvResultsDir = os.path.join(wikiModelDir, 
                     'wiki-detox/src/modeling/cv_results.csv')
    
    bestParams = load_best_params(cvResultsDir,'mlp','char','ed')
    PIPELINE = Pipeline([
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('to_dense', DenseTransformer()), 
                        ('clf', KerasClassifier(build_fn=make_mlp, 
                                                output_dim = 2, 
                                                verbose=False))]) 
    PIPELINE.set_params(**bestParams)
    
    # train models
    trainData = load_training_data(trainDataDir)
    
    attackModel = PIPELINE
    aggrModel = PIPELINE
    
    attackModel.fit(trainData['attackTrainData']['X'],
                    trainData['attackTrainData']['y'])
    aggrModel.fit(trainData['aggrTrainData']['X'],
                    trainData['aggrTrainData']['y'])

    return {
        'attackModel': attackModel,
        'aggrModel': aggrModel
    }


def load_best_params(cv_results_dir, model_type, ngram_type, label_type):
    '''
    Input:
    ======
    cv_result_dir: the directory to "cv_result" file of WikiMedia model
    '''
                               
    import json
    
    cv_results = pd.read_csv(cv_results_dir)
    query = """model_type == \'%s\' and 
                ngram_type == \'%s\' and 
                label_type == \'%s\'""" %(
                model_type, ngram_type, label_type)
        
    params = cv_results.query(query)
    params = params.loc[:,'best_params'].iloc[0]
    return json.loads(params)


def load_training_data(trainDataDir):
    assert os.path.exists(trainDataDir), 'trainDataDir Not Exist'
    attackTrainData = LoadData.load_and_parse_training(trainDataDir,
                                                       'attack',
                                                       'empirical')
    aggrTrainData = LoadData.load_and_parse_training(trainDataDir,
                                                     'aggression',
                                                     'empirical')
    return {
        'attackTrainData': {
                              'X': attackTrainData[0],
                              'y': attackTrainData[1]
                            },
        'aggrTrainData':   {
                              'X': aggrTrainData[0],
                              'y': aggrTrainData[1]
                            }
    }
                               


def diff_clean_text(data):
    ''' Taking the difference between two history version '''
    
    Differ = difflib.Differ()
    ADDED = []
    ADDED_bytes = []
    DELETED = []
    DELETED_bytes = []
    
    for row in data.iterrows():
        # row = [ [idx] [content] ]
        content = row[1]
        assert 'title' in content.keys(), 'Data Format Error, now Title'
        assert 'text' in content.keys(), 'Data Format Error, now Text'

        title = content['title']
        cleaned_text = clean_text(content['text'])


        try:
            if(title == title_prev):
                diff = list(Differ.compare(
                            cleaned_text_prev.split(), 
                            cleaned_text.split()
                            ))

                add_words = []
                del_words = []
                for d in diff:
                    if(d[0] == '+'):
                        add_words.append(d[2:])
                    elif(d[0] == '-'):
                        del_words.append(d[2:])
                added = ' '.join(add_words)
                deleted = ' '.join(del_words)

            else:
                # at the beginning of new page
                added = cleaned_text
                deleted = ''

        except NameError:
            # at the beginning of the file
            added = cleaned_text
            deleted = ''

        ADDED.append(added)
        ADDED_bytes.append(len(added))
        DELETED.append(deleted)
        DELETED_bytes.append(len(deleted))

        title_prev = title
        cleaned_text_prev = cleaned_text
        
    data['Added'] = ADDED
    data['Added_Bytes'] = ADDED_bytes
    data['Deleted'] = DELETED
    data['Deleted_Bytes'] = DELETED_bytes
    
    return data


def clean_text(text):
    ''' 
        Clean the text using the WikiMedia functions 
        but modified to avoid deleting empty rows
    '''
    
    try:
        text = CleanTextData.remove_date(text)
        text = CleanTextData.substitute_patterns(text, CleanTextData.pre_sub_patterns)
        text = CleanTextData.strip_mw(text)
        text = CleanTextData.strip_html(text)
        text = CleanTextData.substitute_patterns(text, CleanTextData.post_sub_patterns)
    
    except TypeError:
        # the string is empty
        text = ''
        
    return text
                               
    
def apply_models_DF(data, model_name, model_dict, cleaned_text_cols):
    ''' Predict the probability of input data to be labelled
        'aggressive' or 'attack' using 
        
        Return:
        =======
        a data frame with pred scores attached
        
    '''
    for col in cleaned_text_cols:
        texts = data[col]
        for task, model in model_dict.items():
            scores = model.predict_proba(texts)[:,1]
            data['%s_%s_%s_score'%(col, task, model_name)] = scores

        return data

def apply_models_text(text, model_dict):
    ''' Predict the probability of input texts to be labelled
        'aggressive' or 'attack'    
        
        Used for sanity check
    '''

    for task,model in model_dict.items():
        scores = model.predict_proba([text])[:,1]
        print('%s_score: %f'%(task,scores))
    
    
    
    
if __name__ == '__main__':
    main()
