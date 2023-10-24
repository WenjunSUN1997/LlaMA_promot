import torch
from model_config.baseline_llame_crf import BaselineLllama

class BaselineLinear(BaselineLllama):

    def __init__(self,
                 model_name,
                 num_label,
                 drop_out,
                 sim_dim,
                 loss_func,
                 no_pad,
                 general):
        super(BaselineLinear, self).__init__(model_name,
                                             num_label,
                                             drop_out,
                                             sim_dim,
                                             no_pad,
                                             general)
        self.loss_func = loss_func

    def forward_sentence(self, data, goal):
        path_all = []
        loss_all = []
        batch_size = data['word_ids'].shape[0]
        first_token_index = self.get_first_token_index(data)
        for batch_index in range(batch_size):
            if self.general:
                label = data['label_general'][batch_index][
                    data['label_general'][batch_index] != -1]
            else:
                label = data['label'][batch_index][
                    data['label'][batch_index] != -1]

            if self.no_pad:
                real_input_ids_index = torch.nonzero(data['attention_mask'][batch_index]).squeeze(-1)
                input_ids = data['input_ids'][batch_index][real_input_ids_index]
                attention_mask = data['attention_mask'][batch_index][real_input_ids_index]
            else:
                input_ids = data['input_ids'][batch_index]
                attention_mask = data['attention_mask'][batch_index]

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            all_token_embedding = self.back_bone_model(input_ids=input_ids,
                                                       attention_mask=attention_mask)['last_hidden_state']
            assert len(label) == len(first_token_index[batch_index])
            first_token_embedding = all_token_embedding[:, first_token_index[batch_index], :]
            if goal == 'train':
                ouput_linear = self.linear(self.drop_out(first_token_embedding))
            else:
                ouput_linear = self.linear(first_token_embedding)

            output_softmax = torch.softmax(ouput_linear, dim=-1)
            path = torch.max(output_softmax, dim=-1).indices[0]
            loss = self.loss_func(output_softmax.squeeze(0), label)
            path_all.append(path.cpu().numpy().tolist())
            loss_all.append(loss)

        return {'path': path_all,
                'loss': sum(loss_all)}
