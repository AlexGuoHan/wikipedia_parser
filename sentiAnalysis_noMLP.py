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
# from keras.wrappers.scikit_learn import KerasClassifier


# import LoadData
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
    # parser.add_argument('--trainDataDir', type=str, 
    #                     dest='trainDataDir', 
    #                     help='directory to training data', 
    #                     required=True)
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
    # trainDataDir = args.trainDataDir
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
    # global mlpModel 
    # mlpModel = load_mlp_char_model(wikiModelDir, trainDataDir)
    
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
    if(num_cpus <= 0): num_cpus = cpu_count()
    print('CPU: %d' % num_cpus)
    
    pool = Pool(num_cpus)
    pool.map(load_apply_save,dataFiles)



    

def load_apply_save(dataDir):
    
    # load data
    tic.go('LOADING & CLEANING DATA %s'%(dataDir))
    raw_data = pd.read_csv(dataDir, sep='\t')
    [cleaned_data, cleaned_text_col] = clean_and_diff(raw_data, verbose=True)
    tic.stop()
    
    # apply two models
    tic.go('APPLYING Logistic MODELS')
    cleaned_data = apply_models_DF(cleaned_data, 
                                   'logistic',
                                   logisticModel, 
                                   cleaned_text_col)
    tic.stop()
    # tic.go('APPLYING MLP MODELS')
    # cleaned_data = apply_models_DF(cleaned_data, 
    #                                'mlp', 
    #                                mlpModel, 
    #                                cleaned_text_col)
    # tic.stop()
    
    # save
    filename = os.path.basename(dataDir)
    cleaned_data.to_csv('predicted_%s'%filename, sep='\t')
    
    
    
    

def load_modules(wikiModelDir):
    ''' This function will import modules based on wmModeiDir variable'''
    assert os.path.exists(wikiModelDir), 'wikiModelDir Not Exist'
    
    global ngram
    global load_comments_and_labels, assemble_data, one_hot
    # global make_mlp, DenseTransformer
    global save_pipeline, load_pipeline
    
    sys.path.append(os.path.join(wikiModelDir,u'wiki-detox/src/modeling'))
    sys.path.append(os.path.join(wikiModelDir,u'wiki-detox/src/data_generation'))
    import ngram
    from baselines import load_comments_and_labels, assemble_data, one_hot
    # from deep_learning import make_mlp, DenseTransformer
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
    query = "model_type == \'%s\' and ngram_type == \'%s\' and label_type == \'%s\'" % (
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
                               

def get_diff(old, new, char_threshold = 5, ratio_threshold = 0.5):
    ''' find diff using exhaustive search, not recommemded '''
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
    return '\n'.join(diff)


def clean_and_diff(data, method='quick_2', verbose=False):
    ''' taking the diff and clean the text column
    
    Return:
    =======
    data: a DataFrame with the cleaned text on 'clean_text' column
    
    '''
    
    # Clean the data
    assert 'title' in data.columns.tolist(), 'DataFrame format Incorrect'
    assert 'text' in data.columns.tolist(), 'DataFrame format Incorrect'
    
    # use wikipedia's clean text data function
    data = CleanTextData.clean_and_filter(data, text_col='text', min_words=0,  min_chars=0, exclude_tokens=False)
    
    # their function will produce some columns we dont need
    data['clean_text'] = data['clean_diff']
    data = data.drop(['diff','clean_diff'],1)
    
    assert 'diff' not in data.columns.tolist()
    assert 'clean_diff' not in data.columns.tolist()
    
    
    titles = data.title.unique()
    idx = 0
    
    # taking the diff for each title
    # delta_byteS = a list for all delta bytes
    # delta_byte = delta byte for this example
    text_diffs = []
    delta_bytes = []
    
    for title in titles:
        
        data_subset = data[data.title == title]
        
        try:
            text_diff = data.clean_text[idx]
            delta_byte = len(data.clean_text[idx])
            
            text_diffs.append(text_diff)
            delta_bytes.append(delta_byte)
            
        except KeyError:
            pass
        
        idx = idx + 1
        
        for idx in range(idx, idx + data_subset.index[-1] - 1):
            
            
            # the clean_and_filter() will delete rows that have 
            #     empty entry after cleaning.
            #     This means that the row provides no information
            #     so leave it empty is okay
            
            
            # to handle this, following rules are adopted
            #     1. if new is empty, but old is not
            #        just skip that row
            #     2. if the new not empty, but old is
            #        not skip
            
            
            try: # test if new is empty
                new = data.clean_text[idx]
            except KeyError:
                if(verbose == True): # the new is empty, skip it
                    print('New is Empty, skipped, at %d'%idx)
                continue
            
            
            try: # test if old is empty
                old = data.clean_text[idx - 1]
            except KeyError:
                # old is empty
                # dont skip it, but make the old empty string
                if(verbose == True):
                    print('Old is EMPTY %d'%idx)
                old = ''
                
            # handle some exceptions
            
            if(type(new) is not str):
                if(verbose == True):
                    print("text is not str: %s, changed to empty"%(new))
                continue
            if(type(old) is not str):
                if(verbose == True):
                    print("text is not str: %s, changed to empty"%(old))
                old = ''                
            
            # assert len(new) > 0
            assert type(text_diff) is str
            

            delta_byte = len(new) - len(old)
            
            if(delta_byte < 0):
                # if delta byte < 0, part of texts has been DELETED
                # append a EMPTY STR
                
                # note text_diffS = list for all text_diff
                # text_diff is the diff for this example
                text_diff = 'DELETED'
            else:
                # get the newly appended textx
                # here ignore the (possibly) deleted texts for simplicity
                if(method == 'quick_2'): 
                    text_diff = new[len(old):]
                
            
                # slow has better performance
                #     quick works okay, but definitely need improvement
                #     slow, quick_1 has bugs that left unsolved
                #     uncomment them if need to use
                #
                # if(method == 'slow'): 
                #     text_diff.append(get_diff(old,new))
                # if(method == 'quick_1'): 
                #     text_diff.append(new.replace(old,' ',1))
            
            # update the lists
           
            
            delta_bytes.append(delta_byte)
            text_diffs.append(text_diff)
            
        
        idx = idx + 1
    
    for key, value in enumerate(data.index.tolist()):
        try:
            data.loc[value, 'diff_clean_text'] = text_diffs[key]
            data.loc[value,'delta_bytes'] = delta_bytes[key]
        except IndexError:
            print(len(text_diffs), len(data.index.tolist()))
            raise
    
    return data, 'diff_clean_text'
                               
    
def apply_models_DF(df, model_name, model_dict, cleaned_text_col):
    ''' Predict the probability of input data to be labelled
        'aggressive' or 'attack' using 
        
        Return:
        =======
        a data frame with pred scores attached
        
    '''
    
    texts = df[cleaned_text_col]
    for task,model in model_dict.items():
        scores = model.predict_proba(texts)[:,1]
        df['%s_%s_score'%(task, model_name)] = scores
    return df

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
