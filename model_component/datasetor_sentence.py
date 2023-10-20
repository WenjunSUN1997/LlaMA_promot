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
                 model_name='bert-base-uncased',
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
        self.max_word_num = max_word_num
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
        for index in range(len(self.csv)):
            result_text_unit.append(self.csv['TOKEN'][index])
            result_label_unit.append(self.csv['label'][index])
            if 'EndOfSentence' in self.csv['MISC'][index]:
                if len(result_text_unit) > self.max_word_num:
                    text_split = self.split_list(result_text_unit, self.max_word_num)
                    label_split = self.split_list(result_label_unit, self.max_word_num)
                    for index in range(len(text_split)):
                        result_text.append(text_split[index])
                        result_label.append(label_split[index])

                else:
                    result_text.append(result_text_unit)
                    result_label.append(result_label_unit)

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
        return {'word_ids': word_ids,
                'input_ids': output_tokenizer['input_ids'].squeeze(0),
                'attention_mask': output_tokenizer['attention_mask'].squeeze(0),
                'label_attention': torch.tensor(label_attention).to(self.device),
                'label': torch.tensor(label_padded).to(self.device)
                }

if __name__ == "__main__":
    file_path_list = ['../data/HIPE-2022-data/data/v2.1/newseye/de/HIPE-2022-v2.1-newseye-train-de.tsv']
    data = read_data(file_path_list=file_path_list)
    datasetor_obj = DatasetorSentence(data['file_list'][0],
                                      label_index_dict=data['label_index_dict'],
                                      index_label_dict=data['index_label_dict']
                                      )
    dataloader = DataLoader(datasetor_obj,
                            batch_size=2,
                            shuffle=False)
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        print(step)