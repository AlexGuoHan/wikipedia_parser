from __future__ import absolute_import
from bs4 import BeautifulSoup
import  mwparserfromhell
import re
import pandas as pd
import copy
import time


### Diff Cleaning ###

months = [u'January',
          u'February',
          u'March',
          u'April',
          u'May',
          u'June',
          u'July',
          u'August',
          u'September',
          u'October',
          u'November',
          u'December',
          u'Jan',
          u'Feb',
          u'Mar',
          u'Apr',
          u'May',
          u'Jun',
          u'Jul',
          u'Aug',
          u'Sep',
          u'Oct',
          u'Nov',
          u'Dec',
        ]


month_or = u'|'.join(months)
date_p = re.compile(u'\d\d:\d\d,( \d?\d)? (%s)( \d?\d)?,? \d\d\d\d (\(UTC\))?' % month_or)
    
def remove_date(comment):
    return re.sub(date_p , u'', comment )



pre_sub_patterns = [
                    (u'\[\[Image:.*?\]\]', u''),
                    (u'<!-- {{blocked}} -->', u''),
                    (u'\[\[File:.*?\]\]', u''),
                    (u'\[\[User:.*?\]\]', u''),
                    (u'\[\[user:.*?\]\]', u''),
                    (u'\(?\[\[User talk:.*?\]\]\)?', u''),
                    (u'\(?\[\[user talk:.*?\]\]\)?', u''),
                    (u'\(?\[\[User Talk:.*?\]\]\)?', u''),
                    (u'\(?\[\[User_talk:.*?\]\]\)?', u''),
                    (u'\(?\[\[user_talk:.*?\]\]\)?', u''),
                    (u'\(?\[\[User_Talk:.*?\]\]\)?', u''),
                    (u'\(?\[\[Special:Contributions.*?\]\]\)?', u''),
                    (u'\|.*?\|',u''),
                   ]

post_sub_patterns = [
                    (u'--', u''),
                    (u' :', u' '),
                    ]

def substitute_patterns(s, sub_patterns):
    for p, r in sub_patterns:
        s = re.sub(p, r, s)
    return s

def strip_html(s):
    try:
        s = BeautifulSoup(s, u'html.parser').get_text()
    except:
        pass
        #print('BS4 HTML PARSER FAILED ON:', s)
    return s

def strip_mw(s):
    try:
        s = mwparserfromhell.parse(s).strip_code()
    except:
        pass
    return s


def clean_comment(s):
    s = remove_date(s)
    s = substitute_patterns(s, pre_sub_patterns)
    s = strip_mw(s)
    s = strip_html(s)
    s = substitute_patterns(s, post_sub_patterns)
    return s


def clean(df):
    df = copy.deepcopy(df)
    df.rename(columns = {u'insertion': u'diff'}, inplace = True)
    df.dropna(subset = [u'diff'], inplace = True)
    df[u'clean_diff'] = df[u'diff']
    df[u'clean_diff'] = df[u'clean_diff'].apply(remove_date)
    df[u'clean_diff'] = df[u'clean_diff'].apply(lambda x: substitute_patterns(x, pre_sub_patterns))
    df[u'clean_diff'] = df[u'clean_diff'].apply(strip_mw)
    df[u'clean_diff'] = df[u'clean_diff'].apply(strip_html)
    df[u'clean_diff'] = df[u'clean_diff'].apply(lambda x: substitute_patterns(x, post_sub_patterns))

    try:
        del df[u'rank']
    except:
        pass
    df.dropna(subset = [u'clean_diff'], inplace = True)
    if not df.empty:
        df = df[df[u'clean_diff'] != u'']
    return df



def show_comments(d, n = 10):
    for i, r in d[:n].iterrows():
        print r[u'diff']
        print u'_' * 80
        print r[u'clean_diff']
        print u'\n\n', u'#' * 80, u'\n\n'


### Admin Filtering ###
# Currently done in HIVE

def find_pattern(d, pattern, column):
    p = re.compile(pattern)
    return d[d[column].apply(lambda x: p.search(x) is not None)]

def exclude_pattern(d, pattern, column):
    p = re.compile(pattern)
    return d[ d[column].apply(lambda x: p.search(x) is None)]

def exclude_few_tokens(d, n):
    return d[d[u'clean_diff'].apply(lambda x:  len(x.split(u' ')) > n)]

def exclude_short_strings(d, n):
    return d[d[u'clean_diff'].apply(lambda x:  len(x) > n)]  

def remove_admin(d, patterns):
    d_reduced = copy.deepcopy(d)
    for pattern in patterns:
        d_reduced = exclude_pattern(d_reduced, pattern, u'diff')
    return d_reduced

patterns =[
    u'\[\[Image:Octagon-warning',
    u'\[\[Image:Stop',
    u'\[\[Image:Information.',
    u'\[\[Image:Copyright-problem',
    u'\[\[Image:Ambox',
    u'\[\[Image:Broom',
    u'\[\[File:Information',
    u'\[\[File:AFC-Logo_Decline',
    u'\[\[File:Ambox',
    u'\[\[File:Nuvola',
    u'\[\[File:Stop',
    u'\[\[File:Copyright-problem',
    u'\|alt=Warning icon\]\]',
    u'The article .* has been \[\[Wikipedia:Proposed deletion\|proposed for deletion\]\]',
    u'Your submission at \[\[Wikipedia:Articles for creation\|Articles for creation\]\]',
    u'A file that you uploaded or altered, .*, has been listed at \[\[Wikipedia:Possibly unfree files\]\]',
    u'User:SuggestBot',
    u'\[\[Wikipedia:Criteria for speedy deletion\|Speedy deletion\]\] nomination of',
    u"Please stop your \[\[Wikipedia:Disruptive editing\|disruptive editing\]\]. If you continue to \[\[Wikipedia:Vandalism\|vandalize\]\] Wikipedia, as you did to .*, you may be \[\[Wikipedia:Blocking policy\|blocked from editing\]\]",
    u"Hello.*and.*\[\[Project:Introduction\|welcome\]\].* to Wikipedia!",
    u'Nomination of .* for deletion',
    u'==.*Welcome.*==',
    u'== 5 Million: We celebrate your contribution ==',
    u'==.*listed for discussion ==',
    ]



def clean_and_filter(df, text_col, min_words=3, min_chars=20):
    t1 = time.time()
    #print('Raw:', df.shape[0])
    df[u'insertion'] = df[text_col]
    df = clean(df).dropna(subset = [u'clean_diff'])
    if df.empty:
        return df
    #print('Cleaned: ', df.shape[0])
    df = exclude_few_tokens(df, min_words)
    #print('No Few Words: ', df.shape[0])
    df = exclude_short_strings(df, min_chars)
    #print('No Few Chars: ', df.shape[0])
    t2 = time.time()
    #print('Cleaning and Filtering Time:',(t2-t1) / 60.0)
    return df

### Data Viz ###
def print_block_data(r):
    block_reasons = r[u'block_reasons'].split(u'PIPE')
    block_timestamps = r[u'block_timestamps'].split(u'PIPE')
    block_actions = r[u'block_actions'].split(u'PIPE')
    block_params = r[u'block_params'].split(u'PIPE')
    
    for i in xrange(len(block_reasons)):
        print u'Log Event #: ', i+1
        print u'Action:',block_actions[i]
        print u'Time:', block_timestamps[i]
        print u'Reason:',block_reasons[i]
        print u'Parameters:',block_params[i]

def print_user_history(d, user):
    
    u"""
    Print out users comments in order
    """
    d = d.fillna(u'')
    
    d_u = d[d[u'user_text'] == user].sort_values(by =u'rev_timestamp')
    
    print u'#' * 80
    print u'History for user: ', user
    print u'\nBlock History'
    print_block_data(d_u.iloc[0])
    print u'#' * 80
    
    
    for i, r in d_u.iterrows():
        print u'\n'
        print u'User: ', r[u'user_text']
        print u'User Talk Page: ', r[u'page_title']
        print u'Timestamp: ', r[u'rev_timestamp']
        print u'\n'
        print r[u'clean_diff']
        print u'_' * 80