import pandas as pd
from tqdm import tqdm
import os
from model_component.clef_evaluation import get_results

def validate(dataloader,
             goal,
             model,
             index_label_dict,
             truth_df,
             lang,
             epoch_num,
             model_name,
             type,
             general,
             index_label_general_dict,
             log_dir):
    predict_df = truth_df.copy(deep=True)
    loss_list = []
    result_index = []
    print('\n' + goal + '\n')
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # break
        output = model(data, 'no_train')
        loss_list.append(output['loss'].item())
        for path_unit in output['path']:
            result_index += path_unit

    if general:
        label_result = sub_process(index_label_general_dict, result_index[:len(truth_df)])
    else:
        label_result = [index_label_dict[x] for x in result_index[:len(truth_df)]]
    # label_result = ['O'] * len(truth_df)
    predict_df['NE-COARSE-LIT'] = label_result
    if not os.path.exists('record/' + log_dir + '/'):
        os.mkdir('record/' + log_dir + '/')

    if not os.path.exists('record/' + log_dir + '/' + type + '/'):
        os.mkdir('record/' + log_dir + '/' + type + '/')

    restore_path_general = 'record/' + log_dir + '/' + type + '/' + model_name.split('/')[-1] + '/'
    if not os.path.exists(restore_path_general):
        os.mkdir(restore_path_general)

    restore_path_specific = restore_path_general + str(epoch_num) + '/'
    if not os.path.exists(restore_path_specific):
        os.mkdir(restore_path_specific)

    truth_df.to_csv(restore_path_specific+'truth.tsv', sep='\t', index=False, encoding='utf8')
    predict_df.to_csv(restore_path_specific+goal+'.tsv', sep='\t', index=False, encoding='utf8')
    get_results(f_ref=restore_path_specific+'truth.tsv',
                f_pred=restore_path_specific+goal+'.tsv',
                outdir=restore_path_specific)
    result = pd.read_csv(restore_path_specific + goal + '_nerc_coarse.tsv', sep='\t')
    fuzzy_f1 = result['F1'][0]
    return {'loss': sum(loss_list) / len(loss_list),
            'result_index': result_index,
            'fuzzy_f1': fuzzy_f1}

def sub_process(index_label_general_dict, predict:list):
    label_index_general_dict = {v: k for k, v in index_label_general_dict.items()}
    index_o = label_index_general_dict['O']
    result = []
    if predict[0] == index_o:
        result.append('O')
    else:
        result.append('B-'+index_label_general_dict[predict[0]])

    for index in range(1, len(predict)):
        if predict[index] == index_o:
            result.append('O')
        else:
            if predict[index] != predict[index-1]:
                result.append('B-'+index_label_general_dict[predict[index]])
            else:
                result.append('I-' + index_label_general_dict[predict[index]])

    return result



if __name__ == "__main__":
    get_results(f_ref='../record/newseye_de/0/truth.tsv',
                f_pred='../record/newseye_de/0/test.tsv',
                outdir='../record/newseye_de/0/')