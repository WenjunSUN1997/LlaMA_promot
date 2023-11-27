# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import json
import numpy as np
import evaluate
from model_component.dataloader import *
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset, DatasetDict
from unmasked_llama.modeling_llama_ori import UnmaskingLlamaForTokenClassification
from model_component.clef_evaluation import get_results
import pandas as pd

device = 'cuda:0'

def get_data_dict(dataset):
    tokens = []
    labels = []
    for index in range(len(dataset.text_sentence)):
        tokens.append(dataset.text_sentence[index])
        labels.append(dataset.label_sentence[index])

    return {'tokens': tokens,
            'ner_tags': labels}

def get_dataset_hf(lang='newseye_fi',
                   model_name='meta-llama/Llama-2-7b-hf'):
    dataset_all = DatasetDict()
    dataloader_train, dataloader_dev, dataloader_test, truth_dev, truth_test, label_index_dict, \
    index_label_dict, weight, weight_general, index_label_general_dict = \
        get_dataloader(batch_size=batch_size,
                       lang=lang,
                       goal='train',
                       model_name=model_name,
                       window=100,
                       step=1,
                       max_token_num=1024,
                       device=device,
                       max_word_num=1000,
                       extensive_model_name=None,
                       type=None)
    dataset_all['train'] = Dataset.from_dict(get_data_dict(dataloader_train.dataset))
    dataset_all['val'] = Dataset.from_dict(get_data_dict(dataloader_dev.dataset))
    dataset_all['test'] = Dataset.from_dict(get_data_dict(dataloader_test.dataset))
    return (dataset_all, label_index_dict, index_label_dict, weight)


model_id = 'meta-llama/Llama-2-7b-hf'
epochs = 1000
batch_size = 8
learning_rate = 1e-4
max_length = 64
lora_r = 24
ds, label2id, id2label, weight = get_dataset_hf()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
seqeval = evaluate.load("seqeval")
label_list = list(label2id.keys()) # ds["train"].features[f"ner_tags"].feature.names
model = UnmaskingLlamaForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id, weight=weight
).bfloat16()
model.to(device)
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS,
                         inference_mode=False, r=lora_r,
                         lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    truth = pd.read_csv('record/truth.tsv',
                        sep='\t',
                        engine='python',
                        encoding='utf8',
                        quoting=3,
                        skip_blank_lines=False).copy(deep=True)
    predic = pd.read_csv('record/test.tsv',
                        sep='\t',
                        engine='python',
                        encoding='utf8',
                        quoting=3,
                        skip_blank_lines=False).copy(deep=True)


    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    label_all = []
    predic_all = []
    for index in range(len(true_predictions)):
        label_all += true_labels[index]
        predic_all += true_predictions[index]

    truth = truth[:len(label_all)]
    predic = predic[:len(predic_all)]
    truth['NE-COARSE-LIT'] = label_all
    truth.to_csv('record/a.tsv', sep='\t', index=False, encoding='utf8')
    predic['NE-COARSE-LIT'] = predic_all
    predic.to_csv('record/b.tsv', sep='\t', index=False, encoding='utf8')
    get_results(f_ref='record/a.tsv',
                f_pred='record/b.tsv',
                outdir='record/')
    result = pd.read_csv('record/b_nerc_coarse.tsv', sep='\t')
    fuzzy_f1 = result['F1'][0]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        'fuzzy_f1': fuzzy_f1,
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="my_awesome_ds_model",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
