from tqdm import tqdm
import os

def validate(dataloader,
             goal,
             model,
             index_label_dict,
             truth_df,
             lang,
             epoch_num):
    predict_df = truth_df.copy(deep=True)
    loss_list = []
    result_index = []
    print('\n' + goal + '\n')
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        output = model(data)
        loss_list.append(output['loss'].item())
        for path_unit in output['path']:
            result_index += path_unit

    label_result = [index_label_dict[x] for x in result_index[:len(truth_df)]]
    predict_df['NE-COARSE-LIT'] = label_result
    restore_path_general = 'record/' + lang + '/'
    if not os.path.exists(restore_path_general):
        os.mkdir(restore_path_general)
    restore_path_specific = restore_path_general + str(epoch_num) + '\\'
    os.mkdir(restore_path_specific)
    truth_df.to_csv(restore_path_specific+'truth.tsv', sep='\t')
    predict_df.to_csv(restore_path_specific+goal+'.tsv', sep='\t')
    os.system('python HIPE-scorer/clef_evaluation.py --ref ' + restore_path_specific+'truth.tsv ' +
              '--pred ' + restore_path_specific + goal + '.tsv ' + '--task nerc_coarse' +
              '--outdir ' + restore_path_specific + ' --skip-check')

    return {'loss': sum(loss_list) / len(loss_list),
            'result_index': result_index}

if __name__ == "__main__":
    os.system('python ../HIPE-scorer/clef_evaluation.py --ref ../record/newseye_de/0/truth.tsv --pred ../record/newseye_de/0/_dev.tsv --skip-check --task nerc_coarse')