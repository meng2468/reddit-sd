import time

import tools.parser as parser
import tools.processing as processing

# Pull data off reddit and store it
def scrape_reddit():
    data_dir = 'data/'
    subreddits = ['CryptoCurrency', 'ethereum','investing','personalfinance','Buttcoin']
    keywords = ['Ethereum', 'ethereum','eth', 'Ether', 'ether', 'ETH']
    days = 1

    data = parser.pull_comments(subreddits, keywords, days)
    timestamp = int(time.time())
    parser.save_to_csv(data_dir + 'c'+'_'+str(timestamp)+'_'+str(days)+'d', data)

    data = parser.pull_submissions(subreddits, keywords, days)
    timestamp = int(time.time())
    parser.save_to_csv(data_dir + 's'+'_'+str(timestamp)+'_'+str(days)+'d', data)

# Convert scraped comments into sentence-level statements and save them
def process_comments():
    data_dir = 'data/'
    # Change this to an existing file name
    file_name = 'c_1620671309_60d.csv'
    
    data = processing.get_processed_data(data_dir + file_name)
    timestamp = int(time.time())
    parser.save_to_csv(data_dir + 'p_' + file_name, data)

# Convert
if __name__ == '__main__':
    scrape_reddit()
    process_comments()