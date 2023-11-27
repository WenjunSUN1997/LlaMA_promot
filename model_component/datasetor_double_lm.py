from model_component.datasetor_sentence import DatasetorSentence
import torch
from transformers import AutoTokenizer

class DatasetorDoubleLM(DatasetorSentence):
    def __init__(self,
                 csv,
                 window=10,
                 step=9,
                 model_name='bert-base-uncased',
                 extensive_model_name='bert-base-uncased',
                 max_token_num=512,
                 label_index_dict=None,
                 index_label_dict=None,
                 device='cuda:0',
                 max_word_num=100,
                 label_index_general_dict=None,
                 index_label_general_dict=None,
                 ):
        super(DatasetorDoubleLM, self).__init__(csv,
                                                window=window,
                                                step=step,
                                                model_name=model_name,
                                                max_token_num=max_token_num,
                                                label_index_dict=label_index_dict,
                                                index_label_dict=index_label_dict,
                                                device=device,
                                                label_index_general_dict=label_index_general_dict,
                                                index_label_general_dict=index_label_general_dict)
        self.extensive_tokenizer = AutoTokenizer.from_pretrained(extensive_model_name)
        if not self.extensive_tokenizer.pad_token:
            self.extensive_tokenizer.pad_token = self.extensive_tokenizer.eos_token

    def __getitem__(self, item):
        label = self.label_sentence[item]
        label_padded = label + [-1] * (self.max_word_num - len(label))
        label_general = self.label_general_sentence[item]
        label_general_padded = label_general + [-1] * (self.max_word_num - len(label_general))
        label_attention = [1] * len(label) + [0] * (self.max_word_num - len(label))
        output_tokenizer = self.tokenizer([self.text_sentence[item]],
                                          is_split_into_words=True,
                                          padding="max_length",
                                          max_length=self.max_token_num,
                                          truncation=True,
                                          return_tensors='pt')
        for key, value in output_tokenizer.items():
            output_tokenizer[key] = value.to(self.device)

        output_tokenizer_extensive = self.extensive_tokenizer([self.text_sentence[item]],
                                                              is_split_into_words=True,
                                                              padding="max_length",
                                                              max_length=1024,
                                                              truncation=True,
                                                              return_tensors='pt')
        for key, value in output_tokenizer_extensive.items():
            output_tokenizer_extensive[key] = value.to(self.device)

        word_ids = torch.tensor([-100 if element is None else element
                                 for element in output_tokenizer.word_ids()]).to(self.device)
        word_ids_extensive = torch.tensor([-100 if element is None else element
                                          for element in output_tokenizer_extensive.word_ids()]).to(self.device)
        if len(label) - 1 != max(word_ids):
            print(item)

        return {'word_ids': word_ids,
                'word_ids_extensive': word_ids_extensive,
                'input_ids': output_tokenizer['input_ids'].squeeze(0),
                'input_ids_extensive': output_tokenizer_extensive['input_ids'].squeeze(0),
                'attention_mask': output_tokenizer['attention_mask'].squeeze(0),
                'attention_mask_extensive': output_tokenizer_extensive['attention_mask'].squeeze(0),
                'label_attention': torch.tensor(label_attention).to(self.device),
                'label': torch.tensor(label_padded).to(self.device),
                'label_general': torch.tensor(label_general_padded).to(self.device),
                'item': torch.tensor(item).to(self.device)
                }