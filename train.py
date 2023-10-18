from model_component.dataloader import get_dataloader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
from model_config.baseline_llame_crf import BaselineLllama

def train(lang='newseye_de',
          model_name='bert-base-uncased',
          num_label=10,
          window=20,
          step=9,
          max_token_num=512,
          device='cuda:0',
          sim_dim=768,
          batch_size=4,
          drop_out=0.3,
          lr=2e-5):
    epoch_num = 1000
    lr = lr
    dataloader_train = get_dataloader(batch_size=batch_size,
                                      lang=lang,
                                      goal='train',
                                      model_name=model_name,
                                      window=window,
                                      step=step,
                                      max_token_num=max_token_num,
                                      device=device)
    dataloader_dev = get_dataloader(batch_size=batch_size,
                                    lang=lang,
                                    goal='dev',
                                    model_name=model_name,
                                    window=window,
                                    step=step,
                                    max_token_num=max_token_num,
                                    device=device)
    dataloader_test = get_dataloader(batch_size=batch_size,
                                     lang=lang,
                                     goal='test',
                                     model_name=model_name,
                                     window=window,
                                     step=step,
                                     max_token_num=max_token_num,
                                     device=device)
    model = BaselineLllama(model_name=model_name,
                           drop_out=drop_out,
                           num_label=num_label,
                           sim_dim=sim_dim)
    model.to(device)
    model.train()
    for param in model.back_bone_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch_index in range(epoch_num):
        loss_list = []
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            output = model(data)
            loss = output['loss']
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('loss:', sum(loss_list) / len(loss_list))

if __name__ == "__main__":
    train()