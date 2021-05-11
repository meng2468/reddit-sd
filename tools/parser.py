import requests
import time
import pandas as pd
import numpy as np

# Pulls all comments for the given paramaters from reddit
def pull_comments(subreddits=['CryptoCurrency'], keywords=['Bitcoin', 'btc'], days=30):
    start_time = time.time()
    start_utc = int(time.time() - days*24*60*60)
    print('*'*40)
    print('Pulling submissions from the past ' + str(days) +'d')
    print('Subreddits', subreddits)
    print('Keywords', keywords)
    keywords = '|'.join(keywords)
    subreddits = ','.join(subreddits)
    url = 'https://api.pushshift.io/reddit/search/comment/?'
    params = ['q='+keywords,'subreddit='+subreddits,'size=100', 'sort=asc']
    data = []

    while True:
        endpoint = url + '&'.join(params) + '&' + 'after='+str(start_utc)
        print('Calling', endpoint)
        r = requests.get(endpoint)
        try:
            daily_data = r.json()
            if len(daily_data['data']) == 0:
                print('No more data to pull, stopping at', time.strftime('%d.%m.%H:%M',time.localtime(start_utc)))
                break
            data += daily_data['data']
            start_utc = daily_data['data'][-1]['created_utc']
        except:
            print('Ran into an error, retrying')

    print('Pulled', len(data), 'submissions in', time.time() - start_time, 'seconds')
    return data

# Pulls all submissions for the given paramaters from reddit
def pull_submissions(subreddits=['CryptoCurrency'], keywords=['Bitcoin', 'btc'], days=30):
    start_time = time.time()
    start_utc = int(time.time() - days*24*60*60)
    print('*'*40)
    print('Pulling submissions from the past ' + str(days) +'d')
    print('Subreddits', subreddits)
    print('Keywords', keywords)
    keywords = '|'.join(keywords)
    subreddits = ','.join(subreddits)
    url = 'https://api.pushshift.io/reddit/search/submission/?'
    params = ['selftext='+keywords,'subreddit='+subreddits,'size=100', 'sort=asc']
    data = []

    while True:
        endpoint = url + '&'.join(params) + '&' + 'after='+str(start_utc)
        print('Calling', endpoint)
        r = requests.get(endpoint)
        try:
            daily_data = r.json()
            if len(daily_data['data']) == 0:
                print('No more data to pull, stopping at', time.strftime('%d.%m.%H:%M',time.localtime(start_utc)))
                break
            data += daily_data['data']
            start_utc = daily_data['data'][-1]['created_utc']
        except:
            print('Ran into an issue, retrying')

    print('Pulled', len(data), 'submissions in', time.time() - start_time, 'seconds')
    return data

def save_to_csv(file_name, data):
    print('Saving data to', file_name+'.csv')
    df = pd.DataFrame(data)
    df.to_csv(file_name+'.csv', index=False)