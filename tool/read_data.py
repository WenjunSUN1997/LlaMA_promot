import pandas as pd

def read_data(file_path_list:list):
    file_list = []
    label_list = []
    for file_path in file_path_list:
        file = pd.read_csv(file_path,
                           sep='\t',
                           engine='python',
                           quotechar='"',
                           error_bad_lines=False)
        row_note = []
        for index in range(len(file)):
            if '#' == file['TOKEN'][index][0]:
                row_note.append(index)

        file = file.drop(row_note)
        file = file.reset_index(drop=True)
        file_list.append(file)
        for data in file['NE-COARSE-LIT']:
            label_list.append(data)

    label_index_dict = {value: index for index, value in enumerate(list(set(label_list)))}
    index_label_dict = {index: value for index, value in enumerate(list(set(label_list)))}
    for file in file_list:
        label = [label_index_dict[x] for x in file['NE-COARSE-LIT']]
        file['label'] = label

    return {'file_list': file_list,
            'label_index_dict': label_index_dict,
            'index_label_dict': index_label_dict}

if __name__ == "__main__":
    file_path_list = ['../data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-dev-en.tsv']
    read_data(file_path_list=file_path_list)