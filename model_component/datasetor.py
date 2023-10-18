from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tool.read_data import read_data
import torch

class Datasetor(Dataset):
    def __init__(self,
                 csv,
                 window=10,
                 step=9,
                 model_name='bert-base-uncased',
                 max_token_num=512,
                 label_index_dict=None,
                 index_label_dict=None,
                 device='cuda:0'):
        self.text = [x for x in csv['TOKEN']]
        self.label = [x for x in csv['label']]
        self.window = window
        self.step = step
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       pad_token='<pad>')
        self.max_token_num = max_token_num
        self.label_index_dict = label_index_dict
        self.index_label_dict = index_label_dict
        self.text_chunked, self.label_chunked = self.split_list_with_padding(window, step)
        self.device = device

    def split_list_with_padding(self,
                                window,
                                step,
                                padding_value='#'):
        text_result = []
        label_result = []
        for i in range(0, len(self.text), step):
            text_chunk = self.text[i:i + window]
            label_chunk = self.label[i:i + window]
            if len(text_chunk) < window:
                text_chunk.extend([padding_value] * (window - len(text_chunk)))
                label_chunk.extend([self.label_index_dict['O']]
                                   * (window - len(text_chunk)))

            text_result.append(text_chunk)
            label_result.append(label_chunk)

        return (text_result, label_result)

    def __len__(self):
        return len(self.text_chunked)

    def __getitem__(self, item):
        label = torch.tensor(self.label_chunked[item]).to(self.device)
        output_tokenizer = self.tokenizer(self.text_chunked[item],
                                          is_split_into_words=True,
                                          padding="max_length",
                                          max_length=self.max_token_num,
                                          truncation=True,
                                          return_tensors='pt')
        for key, value in output_tokenizer.items():
            output_tokenizer[key] = value.to(self.device)

        word_ids = torch.tensor([-100 if element is None else element
                    for element in output_tokenizer.word_ids()]).to(self.device)
        return {'label': label,
                'word_ids': word_ids,
                'input_ids': output_tokenizer['input_ids'].squeeze(1),
                'attention_mask': output_tokenizer['attention_mask'].squeeze(1)}

if __name__ == "__main__":
    file_path_list = ['../data/HIPE-2022-data/data/v2.1/ajmc/en/HIPE-2022-v2.1-ajmc-dev-en.tsv']
    data = read_data(file_path_list=file_path_list)
    datasetor_obj = Datasetor(data['file_list'][0],
                              label_index_dict=data['label_index_dict'],
                              index_label_dict=data['index_label_dict']
                              )
    print(datasetor_obj[0])

