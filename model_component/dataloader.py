from torch.utils.data.dataloader import DataLoader
from model_component.datasetor import Datasetor
from model_component.datasetor_ate import DatasetorAte
from model_component.datasetor_sentence import DatasetorSentence
from tool.read_data import read_data

def get_dataloader(batch_size,
                   lang,
                   goal,
                   model_name,
                   window,
                   step,
                   max_token_num,
                   device,
                   max_word_num):
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
                                     'test': 'data/HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-test-sv.tsv'},
                      'ajmc_en': {'train': 'data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-train-en.tsv',
                                  'dev': 'data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-dev-en.tsv',
                                  'test': 'data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-test-en.tsv'},
                      'ajmc_de': {'train': 'data/HIPE-2022-data/data/v2.1/ajmc/de/HIPE-2022-v2.1-ajmc-train-de.tsv',
                                  'dev': 'data/HIPE-2022-data/data/v2.1/ajmc/de/HIPE-2022-v2.1-ajmc-dev-de.tsv',
                                  'test': 'data/HIPE-2022-data/data/v2.1/ajmc/de/HIPE-2022-v2.1-ajmc-test-de.tsv'},
                      'ajmc_fr': {'train': 'data/HIPE-2022-data/data/v2.1/ajmc/fr/HIPE-2022-v2.1-ajmc-train-fr.tsv',
                                  'dev': 'data/HIPE-2022-data/data/v2.1/ajmc/fr/HIPE-2022-v2.1-ajmc-dev-fr.tsv',
                                  'test': 'data/HIPE-2022-data/data/v2.1/ajmc/fr/HIPE-2022-v2.1-ajmc-test-fr.tsv'}}
    path_list = list(lang_data_dict[lang].values())
    data_pack = read_data(path_list)
    if goal == 'train':
        csv = data_pack['file_list'][0]
    elif goal == 'dev':
        csv = data_pack['file_list'][1]
    else:
        csv = data_pack['file_list'][-1]

    try:
        datasetor = DatasetorSentence(csv=csv,
                                      window=window,
                                      step=step,
                                      model_name=model_name,
                                      max_token_num=max_token_num,
                                      label_index_dict=data_pack['label_index_dict'],
                                      index_label_dict=data_pack['index_label_dict'],
                                      device=device,
                                      max_word_num=max_word_num,
                                      label_index_general_dict=data_pack['label_index_general_dict'],
                                      index_label_general_dict=data_pack['index_label_general_dict'])
    except:
        datasetor = DatasetorAte(csv=csv,
                                 window=window,
                                 step=step,
                                 model_name=model_name,
                                 max_token_num=max_token_num,
                                 label_index_dict=data_pack['label_index_dict'],
                                 index_label_dict=data_pack['index_label_dict'],
                                 device=device,
                                 max_word_num=max_word_num,
                                 label_index_general_dict=data_pack['label_index_general_dict'],
                                 index_label_general_dict=data_pack['index_label_general_dict'])

    dataloader = DataLoader(datasetor,
                            batch_size=batch_size,
                            shuffle=False)
    return (dataloader,
            csv,
            datasetor.label_index_dict,
            datasetor.index_label_dict,
            datasetor.weight,
            datasetor.weight_general,
            datasetor.index_label_general_dict)