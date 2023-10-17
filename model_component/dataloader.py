from torch.utils.data.dataloader import DataLoader
from model_component.datasetor import Datasetor
import pandas as pd
from tool.read_data import read_data

def get_dataloader(batch_size,
                   lang,
                   goal,
                   model_name,
                   window,
                   step,
                   max_token_num,
                   device):
    lang_data_dict = {'newseye_de': {'train': 'data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-train-de.tsv',
                                     'dev': 'data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-dev-de.tsv',
                                     'test': 'data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-test-de.tsv'},
                      'newseye_fi': {'train': 'data/HIPE-2022-data/data/v2.1/newseye/fi/HIPE-2022-v2.1-newseye-train-fi.tsv',
                                     'dev': 'data/HIPE-2022-data/data/v2.1/newseye/fi/HIPE-2022-v2.1-newseye-dev-fi.tsv',
                                     'test': 'data/HIPE-2022-data/data/v2.1/newseye/fi/HIPE-2022-v2.1-newseye-test-fi.tsv'},
                      'newseye_fr': {'train': 'data/HIPE-2022-data/data/v2.1/newseye/fr/HIPE-2022-v2.1-newseye-train-fr.tsv',
                                     'dev': 'data/HIPE-2022-data/data/v2.1/newseye/fr/HIPE-2022-v2.1-newseye-dev-fr.tsv',
                                     'test': 'data/HIPE-2022-data/data/v2.1/newseye/fr/HIPE-2022-v2.1-newseye-test-fr.tsv'},
                      'newseye_sv': {'train': 'data/HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-train-sv.tsv',
                                     'dev': 'data/HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-dev-sv.tsv',
                                     'test': 'data/HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-test-sv.tsv'}}
    path_list = list(lang_data_dict[lang].values())
    data_pack = read_data(path_list)
    if goal == 'train':
        csv = data_pack['file_list'][0]
    elif goal == 'dev':
        csv = data_pack['file_list'][1]
    else:
        csv = data_pack['file_list'][-1]

    datasetor = Datasetor(csv=csv,
                          window=window,
                          step=step,
                          model_name=model_name,
                          max_token_num=max_token_num,
                          label_index_dict=data_pack['label_index_dict'],
                          index_label_dict=data_pack['index_label_dict'],
                          device=device)
    dataloader = DataLoader(datasetor,
                            batch_size=batch_size,
                            shuffle=False)
    return dataloader