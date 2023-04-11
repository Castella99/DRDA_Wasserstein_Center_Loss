#!/bin/bash

# python -u main.py -e=500 -p=50 -b=64 -l=0.1 -m=1 -n=5 -c=0.5 -k=5 -t=${test} > ./log/log_size_${test}_5fold.log

for n_critics in 5 10
do
    for center in 0.1 0.5 1
    do
        for mu in 0.01 0.1 1 10
        do
            for lambda in 0.001 0.01 0.1 1 10
            do
                for test in 640 480 320 160
                do
                    mkdir output/result_cv_5fold_${lambda}_${mu}_${center}_${n_critics}_${test}
                    python -u main.py -e=500 -p=50 -b=64 -l=${lambda} -m=${mu} -n=${n_critics} -c=${center} -k=5 -t=${test} > ./output/result_cv_5fold_${lambda}_${mu}_${center}_${n_critics}_${test}/train_val_test.log
                done
            done
        done
    done
done
