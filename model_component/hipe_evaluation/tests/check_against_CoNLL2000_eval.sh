perl conlleval.pl -d ,  -o NOEXIST < <(paste <(cut -f1 data/unittest-true_bundle3_de_1.tsv) <(cut -f4 data/unittest-true_bundle3_de_1.tsv) <(cut -f4 data/unittest-pred_bundle3_de_1.tsv) | grep -v '#' | tr ',' 'COMMA' | tr '\t' ',' | tail -n +2)
