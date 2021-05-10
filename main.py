import tools.parser as parser
import time

subreddits = ['CryptoCurrency', 'ethereum','investing','personalfinance','Buttcoin']
keywords = ['Ethereum', 'ethereum','eth', 'Ether', 'ether', 'ETH']
days = 60
timestamp = int(time.time())

data = parser.pull_submissions(subreddits, keywords, days)

parser.save_to_csv('s'+'_'+timestamp+'_'+str(days)+'d', data)