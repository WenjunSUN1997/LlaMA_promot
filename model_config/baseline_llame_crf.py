import torch
from transformers import AutoModel
import torchcrf

class BaselineLllama(torch.nn.Module):
    def __init__(self,
                 model_name,
                 num_label,
                 drop_out,
                 sim_dim):
        super(BaselineLllama, self).__init__()
        self.back_bone_model = AutoModel.from_pretrained(model_name)
        self.crf = torchcrf.CRF(num_tags=num_label,
                                batch_first=True)
        self.linear = torch.nn.Linear(in_features=sim_dim,
                                      out_features=num_label)
        self.drop_out = torch.nn.Dropout(p=drop_out)

    def get_first_token_index(self, data):
        word_ids = data['word_ids']
        result = []
        batch_size, sequence_len = word_ids.shape
        for batch_index in range(batch_size):
            result_unit = []
            for token_index in range(sequence_len):
                if word_ids[batch_index][token_index] < 0:
                    continue
                if word_ids[batch_index][token_index] > word_ids[batch_index][token_index-1]:
                    result_unit.append(token_index)

            result.append(result_unit)

        return result

    def get_first_token_embedding(self, all_embedding, first_token_index):
        result = []
        batch_size = all_embedding.shape[0]
        for batch_index in range(batch_size):
            result.append(all_embedding[batch_index][first_token_index[batch_index]])

        return torch.stack(result)

    def forward_sentence(self, data, goal):
        path_all = []
        loss_all = []
        batch_size = data['word_ids'].shape[0]
        first_token_index = self.get_first_token_index(data)
        for batch_index in range(batch_size):
            label = data['label'][batch_index][data['label'][batch_index] != -1]
            input_ids = data['input_ids'][batch_index].unsqueeze(0)
            attention_mask = data['attention_mask'][batch_index].unsqueeze(0)
            all_token_embedding = self.back_bone_model(input_ids=input_ids,
                                                       attention_mask=attention_mask)['last_hidden_state']
            assert len(label) == len(first_token_index[batch_index])
            first_token_embedding = all_token_embedding[:, first_token_index[batch_index], :]
            if goal == 'train':
                ouput_linear = self.linear(self.drop_out(first_token_embedding))
            else:
                ouput_linear = self.linear(first_token_embedding)

            output_softmax = torch.softmax(ouput_linear, dim=-1)
            crf_path = self.crf.decode(output_softmax)
            loss = -1 * self.crf(output_softmax,
                                 label.unsqueeze(0))
            path_all.append(crf_path[0])
            loss_all.append(loss)

        return {'path': path_all,
                'loss': sum(loss_all)}

    def forward_token(self, data, goal):
        first_token_index = self.get_first_token_index(data)
        all_embedding = self.back_bone_model(input_ids=data['input_ids'],
                                             attention_mask=data['attention_mask'])
        embedding_first_token = self.get_first_token_embedding(all_embedding['last_hidden_state'],
                                                               first_token_index)
        if goal == 'train':
            ouput_linear = self.drop_out(self.linear(embedding_first_token))
        else:
            ouput_linear = self.drop_out(embedding_first_token)

        output_softmax = torch.softmax(ouput_linear, dim=-1)
        crf_path = self.crf.decode(output_softmax)
        loss = self.crf(output_softmax,
                        data['label'])
        return {'path': crf_path,
                'loss': loss * -1}

    def forward(self, data, goal):
        return self.forward_sentence(data, goal)



