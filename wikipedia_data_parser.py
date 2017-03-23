import os
import sys
import mwxml
import subprocess
import pandas as pd
import time
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

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
    

def parser(infile,outfile,namespace,page_titles,limit=None):
    '''
    parse the wikipedia.xml file into csv

    Args:
    =====
    infile:    the file to parse
    outfile:     filename of the output file
    namespace:    namespace to search
    page_title:     (set) titles of required pages
    '''

    i = 0
    dump = mwxml.Dump.from_file(infile)
    title=[]
    byte=[]
    user=[]
    timestamp=[]
    comment=[]
    text=[]

    for page in dump:
        if (page.namespace == namespace):
            if (len(page_titles)>0 and page.title.replace(' ','').lower() not in page_titles):
                continue
            prev=0
            for revision in page:
                
                title.append(page.title)
                
                timestamp.append(revision.timestamp)
                
                if(revision.text!=None):
                    text.append(revision.text)
                    byte.append(len(revision.text)-prev)
                    prev=len(revision.text)
                else:
                    byte.append(0)
                    text.append('')
                
                if (revision.user != None):
                    user.append(revision.user.text)
                else:
                    user.append('')
                    
                if(revision.comment!=None):
                    comment.append(revision.comment)
                else:
                    comment.append('')
        if( limit != None and i >= limit ):
            break
        i = i + 1
    df  = pd.DataFrame({'title':title,
                        'time':timestamp,
                        'user':user,
                        'byte':byte,
                        'text':text,
                        'comment':comment})
    
    df.to_csv(outfile, sep='\t', index=False)


def mapper(filename, remove=False):
    namespace=1
    
    # download data
    base_url = 'https://dumps.wikimedia.org/enwiki/20161201/'
    url = base_url+filename
    tic.go('Downloading {}...'.format(filename))
    subprocess.call(["wget", url])
    
    # decompress data
    tic.stop()
    tic.go('Decompresing...')
    subprocess.call(["7z", "e", filename])
    tic.stop()
    
    # parse
    tic.go('Parsing...')
    infile=filename[:-3]
    outfile=filename + '.tsv'
    parser(infile,outfile,namespace,titles)
    tic.stop()
    
    if(remove == True):
        subprocess.call(["rm", filename])
        subprocess.call(["rm", infile])

        
        
def argParser():
    parser = ArgumentParser()
    parser.add_argument('--pageTitleDir', type=str,
                        dest='pageTitleDir', 
                        help='directory to page titles',
                        default=None)
    parser.add_argument('--removeRawData', type=str, 
                        dest='removeRawData', 
                        help='whether to remove raw data',
                        default=False)
    parser.add_argument('--allDumpTextDir', type=str, 
                        dest='allDumpTextDir', 
                        help='directory to dump file names', 
                        required=True)
    parser.add_argument('--debug', 
                        dest='debug', 
                        action='store_true')
    parser.add_argument('--cpu', type=int,
                        dest='cpu',
                        help='number of cpu to deploy',
                        required=True)
    return parser;

        
def main():
    argparser = argParser()
    args = argparser.parse_args()	

    # load titles
    if(args.pageTitleDir != None):
        assert os.path.exists(args.pageTitleDir)
        titles=set()
        with open('page_titles.txt') as f:
            for l in f:
                titles.add(l.strip('\n"'))
    
    # load dump file names
    assert os.path.exists(args.pageTitleDir)
    dumps=[]
    with open(args.allDumpTextDir) as f:
        for l in f:
            dumps.append(l.strip('\n'))
    
    # parallelize over multiple cpu
    assert cpu_count() >= args.cpu,'more cpu than available'
    pool = Pool(args.cpu)
    pool.map(mapper,dumps)
    pool.join()
    pool.close()

if __name__ == '__main__':
    main()
   
