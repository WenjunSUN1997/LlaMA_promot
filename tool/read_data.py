import pandas as pd
import csv

def read_data(file_path_list:list):
    file_list = []
    label_list = []
    for file_path in file_path_list:
        file = pd.read_csv(file_path,
                           sep='\t',
                           engine='python',
                           encoding='utf8',
                           quoting=3,
                           skip_blank_lines=False)
        file = file[~file['TOKEN'].astype(str).str.startswith('#')]
        file = file[~file['TOKEN'].astype(str).str.startswith('\t')]
        file = file.dropna(axis=0)
        file = file.reset_index(drop=True)
        file_list.append(file)
        for data in file['NE-COARSE-LIT']:
            label_list.append(data)

    label_index_dict = {value: index for index, value in enumerate(list(set(label_list)))}
    index_label_dict = {index: value for index, value in enumerate(list(set(label_list)))}
    label_general = set(x.split('-')[-1] for x in list(set(label_list)))
    label_index_general_dict = {value: index for index, value in enumerate(label_general)}
    index_label_general_dict = {index: value for index, value in enumerate(label_general)}
    for file in file_list:
        label = [label_index_dict[x] for x in file['NE-COARSE-LIT']]
        label_general = [label_index_general_dict[x.split('-')[-1]] for x in file['NE-COARSE-LIT']]
        file['label'] = label
        file['label_general'] = label_general

    return {'file_list': file_list,
            'label_index_dict': label_index_dict,
            'index_label_dict': index_label_dict,
            'label_index_general_dict': label_index_general_dict,
            'index_label_general_dict': index_label_general_dict}

if __name__ == "__main__":
    file_path_list = ['../data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-dev-en.tsv']
    read_data(file_path_list=file_path_list)