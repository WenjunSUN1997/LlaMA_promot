from torch.utils.data.dataloader import DataLoader
from model_component.datasetor import Datasetor

def get_dataloader(batch_size,
                   lang,
                   goal,
                   model_name,
                   control_num,
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
    datasetor = Datasetor()