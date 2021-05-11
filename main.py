import tools.parser as parser
import time

subreddits = ['CryptoCurrency', 'ethereum','investing','personalfinance','Buttcoin']
keywords = ['Ethereum', 'ethereum','eth', 'Ether', 'ether', 'ETH']
days = 180

timestamp = int(time.time())
data = parser.pull_comments(subreddits, keywords, days)
parser.save_to_csv('c'+'_'+str(timestamp)+'_'+str(days)+'d', data)