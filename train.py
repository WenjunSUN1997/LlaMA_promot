from model_component.dataloader import get_dataloader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_component.validator import validate
from tqdm import tqdm
import argparse
from model_config.baseline_llame_crf import BaselineLllama

torch.manual_seed(3407)

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
          lr=2e-5,
          max_word_num=100):
    epoch_num = 1000
    lr = lr
    best_dev_loss = 10000000
    best_test_f1 = 0
    best_dev_f1 = 0
    best_dev_f1_when_test = 0
    best_test_f1_when_dev = 0
    best_dev_epoch = 0
    best_test_epoch = 0
    dataloader_train, _, label_index_dict, index_label_dict = get_dataloader(batch_size=batch_size,
                                                                             lang=lang,
                                                                             goal='train',
                                                                             model_name=model_name,
                                                                             window=window,
                                                                             step=step,
                                                                             max_token_num=max_token_num,
                                                                             device=device,
                                                                             max_word_num=max_word_num)
    dataloader_dev, truth_dev, _, _ = get_dataloader(batch_size=batch_size,
                                                     lang=lang,
                                                     goal='dev',
                                                     model_name=model_name,
                                                     window=window,
                                                     step=window,
                                                     max_token_num=max_token_num,
                                                     device=device,
                                                     max_word_num=max_word_num)
    dataloader_test, truth_test, _, _ = get_dataloader(batch_size=batch_size,
                                                       lang=lang,
                                                       goal='test',
                                                       model_name=model_name,
                                                       window=window,
                                                       step=window,
                                                       max_token_num=max_token_num,
                                                       device=device,
                                                       max_word_num=max_word_num)
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
                                   epoch_num=epoch_index,
                                   model_name=model_name)
        performance_test = validate(model=model,
                                    dataloader=dataloader_test,
                                    goal='test',
                                    index_label_dict=index_label_dict,
                                    truth_df=truth_test,
                                    lang=lang,
                                    epoch_num=epoch_index,
                                    model_name=model_name)
        dev_loss = performance_dev['loss']
        dev_f1 = performance_dev['fuzzy_f1']
        test_loss = performance_test['loss']
        test_f1 = performance_test['fuzzy_f1']
        train_loss = sum(loss_list) / len(loss_list)
        scheduler.step(dev_loss)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dev_epoch = epoch_index
            best_dev_f1 = dev_f1
            best_test_f1_when_dev = test_f1

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_dev_f1_when_test = dev_f1
            best_test_epoch = epoch_index

        log = 'epoch: ' + str(epoch_index) + '\n' \
              + 'current train loss: ' + str(train_loss) + '\n' \
              + 'current dev loss: ' + str(dev_loss) + '\n' \
              + 'current dev f1: ' + str(dev_f1) + '\n' \
              + 'current test loss: ' + str(test_loss) + '\n' \
              + 'current test f1: ' + str(test_f1) + '\n' \
              + 'best dev loss: ' + str(best_dev_loss) \
              + '|| test_f1: ' + str(best_test_f1_when_dev) \
              + 'dev_f1: ' + str(best_dev_f1) \
              + '|| epoch: ' + str(best_dev_epoch) + '\n' \
              + 'best test_f1: ' + str(best_test_f1) \
              + '|| dev_f1: ' + str(best_dev_f1_when_test) \
              + '|| epoch: ' + str(best_test_epoch) + '\n' \
              + '*******************************************************\n'
        print(log)
        with open('record/' + lang + '/log.txt', 'w',encoding='utf-8') as file:
            file.write(log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='newseye_sv')
    parser.add_argument("--model_name", default='facebook/bart-large')
    parser.add_argument("--num_label", default=9, type=int)
    parser.add_argument("--window", default=20, type=int)
    parser.add_argument("--max_word_num", default=1000, type=int)
    parser.add_argument("--step", default=10, type=int)
    parser.add_argument("--max_token_num", default=512, type=int)
    parser.add_argument("--sim_dim", default=1024, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--device", default='cuda:0')
    args = parser.parse_args()
    model_name = args.model_name
    lang = args.lang
    num_label = args.num_label
    window = args.window
    max_word_num = args.max_word_num
    step = args.step
    max_token_num = args.max_token_num
    sim_dim = args.sim_dim
    batch_size = args.batch_size
    dropout = args.dropout
    lr = args.lr
    device = args.device
    train(lang=lang,
          model_name=model_name,
          num_label=num_label,
          window=window,
          step=step,
          max_token_num=max_token_num,
          sim_dim=sim_dim,
          batch_size=batch_size,
          drop_out=dropout,
          lr=lr,
          device=device,
          max_word_num=max_word_num)

