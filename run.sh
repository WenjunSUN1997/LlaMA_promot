python train.py  --lang newseye_sv --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1
python train.py  --lang newseye_de --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1
python train.py  --lang newseye_fr --device cuda:2 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1
python train.py  --lang newseye_fi --device cuda:2 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1


python train.py  --lang ann_en --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1 --num_label 2
python train.py  --lang nes_ann --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1 --num_label 2


python train.py  --lang ann_fr --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1 --num_label 2
python train.py  --lang nes_fr --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1 --num_label 2


python train.py  --lang ann_nl --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1 --num_label 2
python train.py  --lang nes_nl --device cuda:4 --model_name meta-llama/Llama-2-7b-chat-hf --max_token_num 2048 --sim_dim 4096 --batch_size 16 --lr 0.01 --type linear --no_pad 1 --num_label 2