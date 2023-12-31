from model_component.dataloader import *
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_component.validator import validate
from tqdm import tqdm
import argparse
from model_config.baseline_llame_crf import BaselineLllama
from model_config.baseline_linear import BaselineLinear
from model_config.baseline_unmask_llama import UnmaskedllamaBaseline
from model_config.double_lm import DoubleLM

torch.manual_seed(3407)

def train(lang='newseye_de',
          model_name='bert-base-uncased',
          extensive_model_name='bert-base-uncased',
          num_label=9,
          window=20,
          step=20,
          max_token_num=512,
          device='cuda:0',
          sim_dim=768,
          batch_size=4,
          drop_out=0.3,
          lr=2e-5,
          max_word_num=100,
          type='linear',
          no_pad=True,
          general=False,
          retrain_backbone=False,
          log_dir='log/',
          unmask_flag=False,
          flag_commu_encoder=True):
    epoch_num = 1000
    lr = lr
    best_dev_loss = 10000000
    best_test_f1 = 0
    best_dev_f1 = 0
    best_dev_f1_when_test = 0
    best_test_f1_when_dev = 0
    best_dev_epoch = 0
    best_test_epoch = 0
    dataloader_train, dataloader_dev, dataloader_test, truth_dev, truth_test, label_index_dict, \
    index_label_dict,  weight, weight_general, index_label_general_dict = \
        get_dataloader(batch_size=batch_size,
                       lang=lang,
                       goal='train',
                       model_name=model_name,
                       window=window,
                       step=step,
                       max_token_num=max_token_num,
                       device=device,
                       max_word_num=max_word_num,
                       extensive_model_name=extensive_model_name,
                       type=type)
    if general:
        loss_func = torch.nn.CrossEntropyLoss(weight=weight_general)
    else:
        loss_func = torch.nn.CrossEntropyLoss(weight=weight)

    if type == 'crf':
        model = BaselineLllama(model_name=model_name,
                               drop_out=drop_out,
                               num_label=num_label,
                               sim_dim=sim_dim,
                               no_pad=no_pad,
                               general=general)
    elif type == 'unllama':
        model = UnmaskedllamaBaseline(model_name=model_name,
                                      loss_func=loss_func,
                                      num_label=num_label,
                                      sim_dim=sim_dim,
                                      weight=weight)
        model.half()
    elif type == 'double_lm':
        model = DoubleLM(expert_model_name=model_name,
                         extensive_model_name=extensive_model_name,
                         loss_func=loss_func,
                         num_label=num_label,
                         sim_dim=sim_dim,
                         drop_out=drop_out,
                         no_pad=no_pad,
                         general=general,
                         unmask=unmask_flag,
                         commu_encoder=flag_commu_encoder)
        for param in model.extensive_model.parameters():
            param.requires_grad = False

    else:
        model = BaselineLinear(model_name=model_name,
                               drop_out=drop_out,
                               num_label=num_label,
                               sim_dim=sim_dim,
                               loss_func=loss_func,
                               no_pad=no_pad,
                               general=general)
    model.to(device)
    model.train()
    if not retrain_backbone:
        for param in model.back_bone_model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=1,
                                  verbose=True)
    for epoch_index in range(epoch_num):
        loss_list = []
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            if step == 10:
                break
            output = model(data, 'train')
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
                                   model_name=model_name,
                                   type=type,
                                   general=general,
                                   index_label_general_dict=index_label_general_dict,
                                   log_dir=log_dir)
        performance_test = validate(model=model,
                                    dataloader=dataloader_test,
                                    goal='test',
                                    index_label_dict=index_label_dict,
                                    truth_df=truth_test,
                                    lang=lang,
                                    epoch_num=epoch_index,
                                    model_name=model_name,
                                    type=type,
                                    general=general,
                                    index_label_general_dict=index_label_general_dict,
                                    log_dir=log_dir)
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
        # path = 'record/' + lang + '/' + type + '/' + model_name.split('/')[-1] + '/log.txt'
        # with open(path, 'w', encoding='utf-8') as file:
        #     file.write(log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='newseye_fi')
    parser.add_argument("--model_name", default='dbmdz/bert-base-historic-multilingual-64k-td-cased')
    parser.add_argument("--extensive_model_name", default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--log_dir", default='log')
    parser.add_argument("--num_label", default=9, type=int)
    parser.add_argument("--window", default=20, type=int)
    parser.add_argument("--max_word_num", default=1000, type=int)
    parser.add_argument("--step", default=10, type=int)
    parser.add_argument("--flag_commu_encoder", default=1, type=int)
    parser.add_argument("--max_token_num", default=512, type=int)
    parser.add_argument("--sim_dim", default=4096, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--unmask_flag", default=0, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--type", default='double_lm', choices=['linear', 'crf',
                                                                'unllama', 'double_lm'])
    parser.add_argument("--device", default='cuda:1')
    parser.add_argument("--no_pad", default=1, type=int)
    parser.add_argument("--retrain_backbone", default=0, type=int)
    parser.add_argument("--general", default=0, type=int)
    args = parser.parse_args()
    print(args)
    general = True if args.general == 1 else False
    log_dir = args.log_dir
    extensive_model_name = args.extensive_model_name
    model_name = args.model_name
    lang = args.lang
    type = args.type
    unmask_flag = True if args.unmask_flag == 1 else False
    retrain_backbone = True if args.retrain_backbone != 0 else False
    no_pad = True if args.no_pad != 0 else False
    num_label = args.num_label
    window = args.window
    max_word_num = args.max_word_num
    flag_commu_encoder = True if args.flag_commu_encoder == 1 else False
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
          max_word_num=max_word_num,
          type=type,
          no_pad=no_pad,
          general=general,
          retrain_backbone=retrain_backbone,
          log_dir=log_dir,
          unmask_flag=unmask_flag,
          extensive_model_name=extensive_model_name,
          flag_commu_encoder=flag_commu_encoder)

