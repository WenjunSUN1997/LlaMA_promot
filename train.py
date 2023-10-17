from model_component.dataloader import get_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse

def train(lang='newseye_de',
          model_name='bert-base-uncased',
          window=10,
          step=9,
          max_token_num=512,
          device='cuda:0',
          sim_dim=768,
          batch_size=4):
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