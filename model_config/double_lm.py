import torch
from model_config.baseline_llame_crf import BaselineLllama
from unmasked_llama.modeling_llama_ori import UnmaskingLlamaModel
from transformers import AutoModel

class DoubleLM(BaselineLllama):
    def __init__(self,
                 expert_model_name,
                 extensive_model_name,
                 num_label,
                 drop_out,
                 sim_dim,
                 loss_func,
                 no_pad,
                 general,
                 unmask,
                 commu_encoder):
        super(DoubleLM, self).__init__(expert_model_name,
                                       num_label,
                                       drop_out,
                                       sim_dim,
                                       no_pad,
                                       general)
        self.back_bone_model = AutoModel.from_pretrained(expert_model_name)
        self.llm_flag = False
        self.flag_commu_encoder = commu_encoder
        self.activation = torch.nn.ReLU()
        if 'llama' in extensive_model_name.lower() \
                or 'mistral' in extensive_model_name.lower():
            self.llm_flag = True
            if unmask:
                self.extensive_model = UnmaskingLlamaModel.from_pretrained(extensive_model_name,
                                                                           torch_dtype=torch.float16)
            else:
                self.extensive_model = AutoModel.from_pretrained(extensive_model_name,
                                                                 torch_dtype=torch.float16)

        else:
            self.extensive_model = AutoModel.from_pretrained(extensive_model_name)

        self.boarden_linear = torch.nn.Linear(in_features=self.back_bone_model.config.hidden_size,
                                              out_features=self.extensive_model.config.hidden_size)
        self.shrink_linear = torch.nn.Linear(in_features=self.extensive_model.config.hidden_size,
                                             out_features=self.back_bone_model.config.hidden_size)
        self.communication_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.extensive_model.config.hidden_size,
                                                                            nhead=4,
                                                                            batch_first=True)
        self.communication_encoder = torch.nn.TransformerEncoder(self.communication_encoder_layer,
                                                                 num_layers=4)
        self.loss_func = loss_func
        self.normal_expert = torch.nn.LayerNorm(self.extensive_model.config.hidden_size)
        self.normal_extensive = torch.nn.LayerNorm(self.extensive_model.config.hidden_size)
        self.normal_all = torch.nn.LayerNorm(self.extensive_model.config.hidden_size)
        self.classifi_linear = torch.nn.Linear(in_features=self.extensive_model.config.hidden_size,
                                               out_features=num_label)

    def get_first_token_index(self, word_ids):
        result = []
        batch_size, sequence_len = word_ids.shape
        for batch_index in range(batch_size):
            result_unit = []
            for token_index in range(sequence_len):
                if word_ids[batch_index][token_index] < 0:
                    continue
                if word_ids[batch_index][token_index] > word_ids[batch_index][token_index - 1]:
                    result_unit.append(token_index)

            result.append(result_unit)

        return result

    def get_first_token_embedding(self, data, part):
        first_token_embedding_list = []
        batch_size = data['word_ids'].shape[0]
        if part == 'expert':
            input_ids_all = data['input_ids']
            word_ids_all = data['word_ids']
            attention_mask_all = data['attention_mask']
        else:
            input_ids_all = data['input_ids_extensive']
            word_ids_all = data['word_ids_extensive']
            attention_mask_all = data['attention_mask_extensive']

        first_token_index = self.get_first_token_index(word_ids_all)
        for batch_index in range(batch_size):
            if self.general:
                label = data['label_general'][batch_index][
                    data['label_general'][batch_index] != -1]
            else:
                label = data['label'][batch_index][
                    data['label'][batch_index] != -1]

            assert len(label) == len(first_token_index[batch_index])
            if self.no_pad:
                real_input_ids_index = torch.nonzero(attention_mask_all[batch_index]).squeeze(-1)
                input_ids = input_ids_all[batch_index][real_input_ids_index]
                attention_mask = attention_mask_all[batch_index][real_input_ids_index]
            else:
                input_ids = input_ids_all[batch_index]
                attention_mask = attention_mask_all[batch_index]

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if part == 'expert':
                all_token_embedding = self.back_bone_model(input_ids=input_ids,
                                                           attention_mask=attention_mask)['last_hidden_state']
            else:
                all_token_embedding = self.extensive_model(input_ids=input_ids,
                                                           attention_mask=attention_mask)['last_hidden_state']

            if self.llm_flag:
                all_token_embedding = all_token_embedding.type(self.linear.weight.dtype)

            first_token_embedding = all_token_embedding[:, first_token_index[batch_index], :]
            first_token_embedding_list.append(first_token_embedding)

        return first_token_embedding_list

    def get_output_encoder(self, embedding_expert_normaled, embedding_extensive_normaled):
        token_num = embedding_expert_normaled.shape[1]
        result = []
        for token_index in range(token_num):
            embedding_to_encode = torch.stack([embedding_expert_normaled[0][token_index],
                                              embedding_extensive_normaled[0][token_index]])
            result.append(torch.mean(self.communication_encoder(embedding_to_encode),
                                     dim=0))

        return torch.stack(result)

    def forward_sentence(self, data, goal):
        path_all = []
        loss_all = []
        batch_size = data['word_ids'].shape[0]
        first_token_embedding_expert = self.get_first_token_embedding(data, 'expert')
        first_token_embedding_extensive = self.get_first_token_embedding(data, 'extensive')
        for batch_size_index in range(batch_size):
            if self.general:
                label = data['label_general'][batch_size_index][
                    data['label_general'][batch_size_index] != -1]
            else:
                label = data['label'][batch_size_index][
                    data['label'][batch_size_index] != -1]

            embedding_expert = self.activation(first_token_embedding_expert[batch_size_index])
            embedding_extensive = self.activation(first_token_embedding_extensive[batch_size_index])
            if goal == 'train':
                embedding_expert = self.drop_out(embedding_expert)

            embedding_expert_boardened = self.boarden_linear(embedding_expert)
            embedding_expert_normaled = self.normal_expert(embedding_expert_boardened)
            embedding_extensive_normaled = self.normal_expert(embedding_extensive)
            if self.flag_commu_encoder:
                final_embedding = self.get_output_encoder(embedding_expert_normaled,
                                                          embedding_extensive_normaled)
            else:
                final_embedding = embedding_expert_normaled + embedding_extensive_normaled

            final_embedding = self.normal_all(final_embedding)
            if goal == 'train':
                final_embedding = self.drop_out(final_embedding)

            output_linear = self.classifi_linear(final_embedding)
            output_softmax = torch.softmax(output_linear, dim=-1)
            path = torch.max(output_softmax, dim=-1).indices
            loss = self.loss_func(output_softmax, label)
            path_all.append(path.cpu().numpy().tolist())
            loss_all.append(loss)

        return {'path': path_all,
                'loss': sum(loss_all)}

