from unmasked_llama.modeling_llama import UnmaskingLlamaForTokenClassification
import torch
from peft import LoraConfig, get_peft_model, TaskType

class UnmaskedllamaBaseline(torch.nn.Module):
    def __init__(self,
                 model_name,
                 loss_func,
                 num_label,
                 sim_dim,
                 weight):
        super(UnmaskedllamaBaseline, self).__init__()
        self.back_bone_model = UnmaskingLlamaForTokenClassification.from_pretrained(model_name,
                                                                                    num_labels=num_label,
                                                                                    torch_dtype=torch.float16)
        self.back_bone_model.loss_weight = weight
        peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS,
                                 inference_mode=False,
                                 r=12,
                                 lora_alpha=32,
                                 lora_dropout=0.1)
        self.back_bone_model = get_peft_model(self.back_bone_model, peft_config)
        self.loss_func = loss_func
        # self.back_bone_model.set_pooling('mean')

    def __forward__(self,
                    word_ids,
                    input_ids,
                    attention_mask,
                    label):
        real_token_index = attention_mask != 0
        input_ids = input_ids[real_token_index]
        attention_mask = attention_mask[real_token_index]
        word_ids = word_ids[real_token_index]
        real_label = label[label != -1]
        label_padded = []
        pre_word_id = -100
        for word_id_unit in word_ids:
            if word_id_unit == -100:
                label_padded.append(-100)
            elif word_id_unit != pre_word_id:
                pre_word_id = word_id_unit
                label_padded.append(real_label[pre_word_id].item())
            else:
                label_padded.append(-100)

        label_padded = torch.tensor(label_padded).unsqueeze(0).to(input_ids.device)
        output_backbone = self.back_bone_model(input_ids=input_ids.unsqueeze(0),
                                               attention_mask=attention_mask.unsqueeze(0),
                                               labels=label_padded)
        return output_backbone

    def forward(self, data, goal):
        batch_size = data['input_ids'].shape[0]
        path_all = []
        loss = []
        for batch_index in range(batch_size):
            input_ids_this_batch = data['input_ids'][batch_index]
            attention_mask_this_batch = data['attention_mask'][batch_index]
            word_ids_this_batch = data['word_ids'][batch_index]
            label_this_batch = data['label'][batch_index]
            output_this_batch = self.__forward__(word_ids_this_batch,
                                                 input_ids_this_batch,
                                                 attention_mask_this_batch,
                                                 label_this_batch)
            loss.append(output_this_batch['loss'])
            path_all.append(output_this_batch['path'])

        return ({'path': path_all,
                'loss': sum(loss)})

