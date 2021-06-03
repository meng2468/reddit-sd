import pandas as pd
import os.path

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df


def authors_selection(file_path_1, file_path_2, encoding='utf-8'):
    #data csv files
    data_1 = read_data(file_path_1)
    data_2 = read_data(file_path_2)
    #outputpath to write to the file
    output_file_path = '../data/selected_authors.csv'
    note_same_annotation_path = '../data/not_same_annotation.csv'
    columns = 'author,label\n'
    with open(note_same_annotation_path, 'a', encoding=encoding) as file:
        file.write(columns)
    
    if os.path.isfile(output_file_path):
        columns = ''
    
    with open(output_file_path, 'a', encoding=encoding) as file:
        file.write(columns)
        
        i = 0
        for author1, stance1 in zip(data_1['id'].values, data_1[' author'].values):

            #check if we all have the same authors
            author_flag = False

            for author2, stance2 in zip(data_2['id'].values, data_2[' author'].values):

                if author1 == author2:
                    author_flag = True

                    if stance1 == stance2:
                        i += 1
                        final_author = author2
                        final_stance = stance2
                        line = final_author + ',' + final_stance + '\n'
                        file.write(line)
                    else:
                        with open(note_same_annotation_path, 'a', encoding=encoding) as file2:
                            line = author2 + '\n'
                            file2.write(line)

                    break

            '''
            add authors that do not exist in both files
            '''
            if author_flag == False:
                print('Author ', author1, ' not found, adding to selection file!!!!')
                final_author = author1
                final_stance = stance1
                line = final_author + ',' + final_stance + '\n'
                file.write(line)

        print('Count of users with same annotated stance', i)

def generate_dataset(author_file_path, comments_file_path, encoding='utf-8'):
    authors_data = read_data(author_file_path)
    comments_data = read_data(comments_file_path)
    output_file_path = '../data/author_comment_dataset.csv'
    columns = 'author,label,target,text\n'
    target = 'Ethereum'

    if os.path.isfile(output_file_path):
        columns = ''
    
    with open(output_file_path, 'a', encoding=encoding) as file:
        file.write(columns)
        for author, stance in zip(authors_data['author'].values, authors_data['label'].values):
            for sentence in comments_data[comments_data.author == author]['text'].values:
                #uncomment this line to replace characters
                #sentence = sentence.replace('"', '\\"')
                line = str(author) + ',' + str(stance) + ',' + str(target) + ',`' + str(sentence) + '`\n'
                file.write(line)

if __name__ == '__main__':
    file_path_1 = '../data/auth_annotations_toghrul&malte.csv'
    file_path_2 = '../data/auth_annotations_malte.csv'

    comments_file_path = '../data/p_c_1620679531_180d.csv'
    selected_authors_path = '../data/selected_authors.csv'



    #UNCOMMENT to generate file with same author annotation
    #authors_selection(file_path_1, file_path_2)
    #UNCOMMENT to generate final dataset
    #generate_dataset(selected_authors_path, comments_file_path)
    #READ DATASET WITH THIS LINE
    df = pd.read_csv('../data/author_comment_dataset.csv', quotechar="`")
    print(df)
