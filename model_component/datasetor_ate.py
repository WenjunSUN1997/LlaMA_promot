from model_component.datasetor_sentence import DatasetorSentence
from tool.read_data import read_data
from tqdm import tqdm

stop_words = [".", "!", "?"]

class DatasetorAte(DatasetorSentence):
    def __init__(self,
                 csv,
                 window=10,
                 step=9,
                 model_name='bert-base-uncased',
                 max_token_num=512,
                 label_index_dict=None,
                 index_label_dict=None,
                 device='cuda:0',
                 max_word_num=100,
                 label_index_general_dict=None,
                 index_label_general_dict=None
                 ):
        super(DatasetorAte, self).__init__(csv,
                                           window=window,
                                           step=step,
                                           model_name=model_name,
                                           max_token_num=max_token_num,
                                           label_index_dict=label_index_dict,
                                           index_label_dict=index_label_dict,
                                           device=device,
                                           label_index_general_dict=label_index_general_dict,
                                           index_label_general_dict=index_label_general_dict)

    def split_by_sentence(self):
        result_text = []
        result_label = []
        result_label_general = []
        result_text_unit = []
        result_label_unit = []
        result_label_general_unit = []
        print('\n prepare data')
        for index in tqdm(range(len(self.csv))):
            result_text_unit.append(self.csv['TOKEN'][index])
            result_label_unit.append(self.csv['label'][index])
            result_label_general_unit.append(self.csv['label_general'][index])
            if self.csv['TOKEN'][index] in stop_words \
                    or index == len(self.csv) - 1:
                text_split = []
                label_split = []
                label_general_split = []
                while True:
                    output_tokenizer = self.tokenizer(result_text_unit,
                                                      is_split_into_words=True,
                                                      padding="max_length",
                                                      max_length=self.max_token_num,
                                                      truncation=True,
                                                      return_tensors='pt')
                    max_word_id = max([-100 if element is None else element
                                       for element in output_tokenizer.word_ids()])
                    if max_word_id < len(result_text_unit) - 1:
                        text_split.append(result_text_unit[:max_word_id])
                        label_split.append(result_label_unit[:max_word_id])
                        label_general_split.append(result_label_general_unit[:max_word_id])
                        result_text_unit = result_text_unit[max_word_id:]
                        result_label_unit = result_label_unit[max_word_id:]
                        result_label_general_unit = result_label_general_unit[max_word_id:]
                    else:
                        text_split.append(result_text_unit)
                        label_split.append(result_label_unit)
                        label_general_split.append(result_label_general_unit)
                        break

                for index_split in range(len(text_split)):
                    result_text.append(text_split[index_split])
                    result_label.append(label_split[index_split])
                    result_label_general.append(label_general_split[index_split])

                result_text_unit = []
                result_label_unit = []
                result_label_general_unit = []

        return (result_text, result_label, result_label_general)

if __name__ == "__main__":
    file_path_list = ['../data/ate/en_ann_train.tsv']
    data = read_data(file_path_list=file_path_list)
    datasetor_obj = DatasetorAte(data['file_list'][0],
                              label_index_dict=data['label_index_dict'],
                              index_label_dict=data['index_label_dict']
                              )
    print(datasetor_obj[0])