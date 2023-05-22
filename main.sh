#!/bin/bash
# python -u main.py -e=500 -p=50 -b=64 -l=0.1 -m=1 -n=5 -c=0.5 -k=5 -t=${test} > ./log/log_size_${test}_5fold.log

# Loop through two values of mu
for mu in 5
do
    # Loop through five values of l
    for l in 1
    do
        args="-e=2000 -p=500 -b=64 -l=${l} -m=${mu} -n=5 -k=5 -t=160"
        instance=e2000_p500_b64_l${l}_m${mu}_n5_k5_t160
        # Check if the result_table.csv file exists
        if test -f "output_${instance}/result_table.csv"; then
        # If the file exists, print "learning Complete"
        echo "learning Complete"
        else 
        # If the file does not exist, create a directory and run a Python file
        mkdir "output_${instance}"
        mkdir "output_${instance}/result_cv_5fold"
        # Print the command to run the Python file
        echo "Run Python File main.py ${args}"
        # Run the Python file and save the output to a log file
        python -u main.py ${args} > ./output_${instance}/train_val_test.log
        fi
    done
done
