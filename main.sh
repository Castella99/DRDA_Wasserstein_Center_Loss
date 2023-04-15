#!/bin/bash
# python -u main.py -e=500 -p=50 -b=64 -l=0.1 -m=1 -n=5 -c=0.5 -k=5 -t=${test} > ./log/log_size_${test}_5fold.log

for mu in 0.01 0.1 1 10
do
    for test in 640 480 320 160
    do
        if [-f output/result_cv_5fold_10_${mu}_${center}_5_${test}/result_table.csv]; then
            echo "Already Complete"
        else 
            mkdir output/result_cv_5fold_${lambda}_${mu}_${center}_${n_critics}_${test}
            echo "Run Python File main.py -e=500 -p=50 -b=64 -l=10 -m=${mu} -n=5 -k=5 -t=${test}"
            python -u main.py -e=500 -p=50 -b=64 -l=10 -m=${mu} -n=5 -k=5 -t=${test} > ./output/result_cv_5fold_${mu}_${test}/train_val_test.log
        fi
    done
done
