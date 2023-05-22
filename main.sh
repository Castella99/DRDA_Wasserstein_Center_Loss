#!/bin/bash
# python -u main.py -e=500 -p=50 -b=64 -l=0.1 -m=1 -n=5 -c=0.5 -k=5 -t=${test} > ./log/log_size_${test}_5fold.log

# Loop through two values of mu
for mu in 5
do
    # Loop through five values of l
    for l in 1
    do
        # Loop through one value of test
        for test in 160 320 480 640
        do
            # Check if the result_table.csv file exists
            if test -f "output_${l}_${mu}/result_cv_5fold_${l}_${mu}_${test}/result_table.csv"; then
            # If the file exists, print "learning Complete"
            echo "learning Complete"
            else 
            # If the file does not exist, create a directory and run a Python file
            mkdir output_${l}_${mu}
            mkdir output_${l}_${mu}/result_cv_5fold_${test}
            # Print the command to run the Python file
            echo "Run Python File main.py -e=2000 -p=500 -b=64 -l=${l} -m=${mu} -n=5 -k=5 -t=${test}"
            # Run the Python file and save the output to a log file
            python -u main.py -e=2000 -p=500 -b=64 -l=${l} -m=${mu} -n=5 -k=5 -t=${test} > ./output_${l}_${mu}/result_cv_5fold_${test}/train_val_test.log
            fi
        done
    done
done
