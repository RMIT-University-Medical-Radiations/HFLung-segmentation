#!/bin/bash

patient_id=1
while [ $patient_id -le 20 ]
    do
        printf -v pstr "%02d" $patient_id
        echo "/datasets/RNSH_HFlung/pre-processed-plastimatch/mha/Patient$pstr"
        ((patient_id++))
    done
echo All done
