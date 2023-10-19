from model_component.dataloader import get_dataloader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_component.validator import validate
from tqdm import tqdm
import argparse
from model_config.baseline_llame_crf import BaselineLllama

def train(lang='newseye_de',
          model_name='bert-base-uncased',
          num_label=9,
          window=20,
          step=20,
          max_token_num=512,
          device='cuda:0',
          sim_dim=768,
          batch_size=4,
          drop_out=0.3,
          lr=2e-5):
    epoch_num = 1000
    lr = lr
    best_dev_loss = 10000000
    best_test_f1 = 0
    best_dev_f1 = 0
    best_dev_f1_when_test = 0
    best_dev_epoch = 0
    best_test_epoch = 0
    dataloader_train, _, label_index_dict, index_label_dict = get_dataloader(batch_size=batch_size,
                                                                             lang=lang,
                                                                             goal='train',
                                                                             model_name=model_name,
                                                                             window=window,
                                                                             step=step,
                                                                             max_token_num=max_token_num,
                                                                             device=device)
    dataloader_dev, truth_dev, _, _ = get_dataloader(batch_size=batch_size,
                                                     lang=lang,
                                                     goal='dev',
                                                     model_name=model_name,
                                                     window=window,
                                                     step=window,
                                                     max_token_num=max_token_num,
                                                     device=device)
    dataloader_test, truth_test, _, _ = get_dataloader(batch_size=batch_size,
                                                       lang=lang,
                                                       goal='test',
                                                       model_name=model_name,
                                                       window=window,
                                                       step=window,
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
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=2,
                                  verbose=True)
    for epoch_index in range(epoch_num):
        loss_list = []
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            output = model(data)
            loss = output['loss']
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break

        print('loss:', sum(loss_list) / len(loss_list))
        performance_dev = validate(model=model,
                                   dataloader=dataloader_dev,
                                   goal='dev',
                                   index_label_dict=index_label_dict,
                                   truth_df=truth_dev,
                                   lang=lang,
                                   epoch_num=epoch_index)
        performance_test = validate(model=model,
                                    dataloader=dataloader_test,
                                    goal='test',
                                    index_label_dict=index_label_dict,
                                    truth_df=truth_test,
                                    lang=lang,
                                    epoch_num=epoch_index)
        dev_loss = performance_dev['loss']
        dev_f1 = performance_dev['f1']
        test_loss = performance_test['loss']
        test_f1 = performance_test['f1']
        train_loss = sum(loss_list) / len(loss_list)
        scheduler.step(dev_loss)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dev_epoch = epoch_index
            best_dev_f1 = dev_f1

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_dev_f1_when_test = dev_f1
            best_test_epoch = epoch_index

        print('epoch: ' + str(epoch_index) + '\n')
        print('current train loss: ' + str(train_loss) + '\n')
        print('current dev loss: ' + str(dev_loss) + '\n')
        print('current dev f1: ' + str(dev_f1) + '\n')
        print('current test loss: ' + str(test_loss) + '\n')
        print('current test f1: ' + str(test_f1) + '\n')
        print('best dev loss: ' + str(best_dev_loss) + '|| test_f1: ' + str(test_f1) +
              'dev_f1: ' + str(best_dev_f1) + '|| epoch: ' + str(best_dev_epoch) + '\n')
        print('best test_f1: ' + str(best_test_f1) + '|| dev_f1: ' + str(best_dev_f1_when_test) +
              '|| epoch: ' + str(best_test_epoch) + '\n')
        print('*******************************************************\n')


if __name__ == "__main__":
    train()