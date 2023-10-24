from model_component.datasetor import Datasetor
from tool.read_data import read_data
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

class DatasetorSentence(Datasetor):

    def __init__(self,
                 csv,
                 window=10,
                 step=9,
                 model_name='mistralai/Mistral-7B-v0.1',
                 max_token_num=512,
                 label_index_dict=None,
                 index_label_dict=None,
                 device='cuda:0',
                 max_word_num=100):
        super(DatasetorSentence, self).__init__(csv,
                                                window=window,
                                                step=step,
                                                model_name=model_name,
                                                max_token_num=max_token_num,
                                                label_index_dict=label_index_dict,
                                                index_label_dict=index_label_dict,
                                                device=device)
        self.max_word_num = 1000
        self.text_sentence, self.label_sentence = self.split_by_sentence()

    def split_list(self, input_list, m):
        result = []
        for i in range(0, len(input_list), m):
            small_list = input_list[i:i + m]
            result.append(small_list)

        return result

    def split_by_sentence(self):
        result_text = []
        result_label = []
        result_text_unit = []
        result_label_unit = []
        print('\n prepare data')
        for index in tqdm(range(len(self.csv))):
            result_text_unit.append(self.csv['TOKEN'][index])
            result_label_unit.append(self.csv['label'][index])
            if 'EndOfSentence' in self.csv['MISC'][index] \
                    or index == len(self.csv)-1:
                text_split = []
                label_split = []
                while True:
                    output_tokenizer = self.tokenizer(result_text_unit,
                                                      is_split_into_words=True,
                                                      padding="max_length",
                                                      max_length=self.max_token_num,
                                                      truncation=True,
                                                      return_tensors='pt')
                    max_word_id = max([-100 if element is None else element
                                      for element in output_tokenizer.word_ids()])
                    if max_word_id < len(result_text_unit)-1:
                        text_split.append(result_text_unit[:max_word_id])
                        label_split.append(result_label_unit[:max_word_id])
                        result_text_unit = result_text_unit[max_word_id:]
                        result_label_unit = result_label_unit[max_word_id:]
                    else:
                        text_split.append(result_text_unit)
                        label_split.append(result_label_unit)
                        break

                for index_split in range(len(text_split)):
                    result_text.append(text_split[index_split])
                    result_label.append(label_split[index_split])

                result_text_unit = []
                result_label_unit = []

        return (result_text, result_label)

    def __len__(self):
        return len(self.text_sentence)

    def __getitem__(self, item):
        label = self.label_sentence[item]
        label_padded = label + [-1] * (self.max_word_num - len(label))
        label_attention = [1] * len(label) + [0] * (self.max_word_num - len(label))
        output_tokenizer = self.tokenizer([self.text_sentence[item]],
                                          is_split_into_words=True,
                                          padding="max_length",
                                          max_length=self.max_token_num,
                                          truncation=True,
                                          return_tensors='pt')
        for key, value in output_tokenizer.items():
            output_tokenizer[key] = value.to(self.device)

        word_ids = torch.tensor([-100 if element is None else element
                                 for element in output_tokenizer.word_ids()]).to(self.device)
        if len(label)-1 != max(word_ids):
            print(item)

        return {'word_ids': word_ids,
                'input_ids': output_tokenizer['input_ids'].squeeze(0),
                'attention_mask': output_tokenizer['attention_mask'].squeeze(0),
                'label_attention': torch.tensor(label_attention).to(self.device),
                'label': torch.tensor(label_padded).to(self.device),
                'item': torch.tensor(item).to(self.device)
                }

def check(datasetor:DatasetorSentence):
    length = [0]
    a = []
    for x in datasetor.text_sentence:
        a += x
    b = [x for x in datasetor.csv['TOKEN']]
    for index in range(len(a)):
        if b[index] != a[index]:
            print(index)

    return b

if __name__ == "__main__":
    file_path_list = ['../data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-dev-de.tsv']
    data = read_data(file_path_list=file_path_list)
    datasetor_obj = DatasetorSentence(data['file_list'][0],
                                      label_index_dict=data['label_index_dict'],
                                      index_label_dict=data['index_label_dict'],
                                      max_word_num=1000
                                      )
    check(datasetor_obj)
    dataloader = DataLoader(datasetor_obj,
                            batch_size=10,
                            shuffle=False)
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if step != 325:
            continue
