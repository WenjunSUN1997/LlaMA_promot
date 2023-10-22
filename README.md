--lang: newseye_[de, fr, fi, sv]

--model_name: the name of language model

--num_label: the number of labels include 'O', for newseye=9

--window: no need to set, will remove

--step:  no need to set, will remove

--max_word_num: no need to set, will remove

--max_token_num: the number of subtokens for tokenizer

--sim_dim: the dimention of the token embedding, for bert=768

--batch_size: defalut=4

--dropout: default=0.3

--lr: learn rate, default=2e-5

--device: default='cuda:0'