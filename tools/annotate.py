import pandas as pd
import os.path

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def annotate(file_path, output_file, encoding='utf-8'):
    data = pd.read_csv(file_path)
    columns = 'id, author, created_utc, score, subreddit, text, label\n'
    if os.path.isfile(output_file):
        columns = ''
    with open(output_file, 'a', encoding="utf-8") as file:
        file.write(columns)
        annotation = input('Type anything to get started   ')
        annotated = 0
        while annotation != 'q':
            row = data.sample()
            print('-'*40)
            print('Annotation', annotated,', Target: Ethereum')
            print('Text:', row['text'].values[0])
            annotation = input('1. (f)avor 2. (a)gainst 3. (n)either or 4. (u)unsure? 5. (q) to exit :  ')
            if len(annotation) == 0:
                print('Empty input, skipping')
                continue
            if annotation[0] in ['f', '1']:
                annotation = 'favor'
            elif annotation[0] in ['a','2']:
                annotation = 'against'
            elif annotation[0] in ['n','3']:
                annotation = 'neither'
            elif annotation[0] in ['u','4']:
                print('Skipping')
                continue
            elif annotation[0] in ['q', '5']:
                print('Quitting')
                break
            else:
                print("Didn't choose any option, skipping")
                continue
            print('Chose', annotation)
            line =  str(row['id'].values[0]) + ', ' + str(row['author'].values[0])+ ', '
            line += str(row['created_utc'].values[0]) + ',' + str(row['score'].values[0]) + ','
            line += str(row['score'].values[0]) + ',"'
            line += str(row['text'].values[0]) + '",' + annotation + '\n'
            file.write(line)
            annotated += 1

if __name__ == '__main__':
    file_path = '../data/p_c_1620679531_180d.csv'
    output_file = '../data/annotations.csv'

    annotate(file_path, output_file)
