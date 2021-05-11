import pandas as pd

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def annotate(file_path, output_file):
    data = pd.read_csv(file_path)
    with open(output_file, 'a') as file:
        annotation = input('Type anything to get started   ')
        annotated = 0
        while annotation != 'q':
            row = data.sample()
            print('-'*40)
            print('Annotation', annotated)
            print('Text:', row['text'].values[0])
            annotation = input('(f)avor, (a)gainst, (n)either, or (u)unsure? (q) to exit:  ')
            if annotation[0] == 'f':
                annotation = 'favor'
            elif annotation[0] == 'a':
                annotation = 'against'
            elif annotation[0] == 'n':
                annotation = 'neither'
            elif annotation[0] == 'u':
                continue
                annotation = 'unsure'
            elif annotation[0] == 'q':
                print('Quitting')
                return
            file.write(str(row['id'].values[0])+ ', "' + row['text'].values[0] + '", ' + annotation)
            annotated += 1

if __name__ == '__main__':
    file_path = '../data/p_c_1620679531_180d.csv'
    output_file = '../data/annotations.csv'

    annotate(file_path, output_file)