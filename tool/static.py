import pandas as pd
from collections import Counter

def static(lang):
    lang_data_dict = {
        'newseye_de': {'train': '../data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-train-de.tsv',
                       'dev': '../data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-dev-de.tsv',
                       'test': '../data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-test-de.tsv'},
        'newseye_fi': {'train': '../data/HIPE-2022-data/data/v2.1/newseye/fi/HIPE-2022-v2.1-newseye-train-fi.tsv',
                       'dev': '../data/HIPE-2022-data/data/v2.1/newseye/fi/HIPE-2022-v2.1-newseye-dev-fi.tsv',
                       'test': '../data/HIPE-2022-data/data/v2.1/newseye/fi/HIPE-2022-v2.1-newseye-test-fi.tsv'},
        'newseye_fr': {'train': '../data/HIPE-2022-data/data/v2.1/newseye/fr/HIPE-2022-v2.1-newseye-train-fr.tsv',
                       'dev': '../data/HIPE-2022-data/data/v2.1/newseye/fr/HIPE-2022-v2.1-newseye-dev-fr.tsv',
                       'test': '../data/HIPE-2022-data/data/v2.1/newseye/fr/HIPE-2022-v2.1-newseye-test-fr.tsv'},
        'newseye_sv': {'train': '../data/HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-train-sv.tsv',
                       'dev': '../data/HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-dev-sv.tsv',
                       'test': '../data/HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-test-sv.tsv'},
        'ajmc_en': {'train': '../data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-train-en.tsv',
                    'dev': '../data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-dev-en.tsv',
                    'test': '../data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-test-en.tsv'},
        'ajmc_de': {'train': '../data/HIPE-2022-data/data/v2.1/ajmc/de/HIPE-2022-v2.1-ajmc-train-de.tsv',
                    'dev': '../data/HIPE-2022-data/data/v2.1/ajmc/de/HIPE-2022-v2.1-ajmc-dev-de.tsv',
                    'test': '../data/HIPE-2022-data/data/v2.1/ajmc/de/HIPE-2022-v2.1-ajmc-test-de.tsv'},
        'ajmc_fr': {'train': '../data/HIPE-2022-data/data/v2.1/ajmc/fr/HIPE-2022-v2.1-ajmc-train-fr.tsv',
                    'dev': '../data/HIPE-2022-data/data/v2.1/ajmc/fr/HIPE-2022-v2.1-ajmc-dev-fr.tsv',
                    'test': '../data/HIPE-2022-data/data/v2.1/ajmc/fr/HIPE-2022-v2.1-ajmc-test-fr.tsv'}}
    target_dict = lang_data_dict[lang]
    path_list = target_dict.values()
    label_list = []
    for path in path_list:
        df = pd.read_csv(path,
                         sep='\t',
                         engine='python',
                         encoding='utf8',
                         quoting=3,
                         skip_blank_lines=False)
        label = [x for x in df['NE-COARSE-LIT']]
        label_list += label

    element_count = Counter(label_list)
    for element, count in element_count.items():
        print(f"{element}: {count}")



if __name__ == "__main__":
    static('newseye_de')