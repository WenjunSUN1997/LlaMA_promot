import torch
from transformers import AutoModel

class BaselineLllama(torch.nn.Module):
    def __init__(self,
                 model_name):
        super(BaselineLllama, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def get_first_token_index(self, data):
        word_ids = data['word_ids']
        result = []
        batch_size, sequence_len = word_ids.shape
        for batch_index in range(batch_size):
            result_unit = []
            for token_index in range(sequence_len):
                if word_ids[batch_index][token_index] <= 0:
                    continue
                if word_ids[batch_index][token_index] > word_ids[batch_index][token_index-1]:
                    result_unit.append(token_index)

            result.append(result_unit)

        return result

    def get_embedding_first_token(self, data):
        all_embedding = self.model(input_ids=data['input_ids'],
                                   attention_mask=data['attention_mask'])
        first_token_index = self.get_first_token_index(data)
        first_token_embedding = all_embedding[first_token_index]

    def forward(self, data):
        embedding_first_token = self.get_embedding_first_token(data)