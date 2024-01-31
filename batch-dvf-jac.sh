#!/bin/bash

patient_id=1
while [ $patient_id -le 20 ]
    do
        printf -v pstr "%02d" $patient_id
        cd "/datasets/RNSH_HFlung/pre-processed-plastimatch/mha/Patient$pstr"
        echo "Processing Patient$pstr"
        plastimatch register /HFLung-segmentation/register-commands.txt
        plastimatch jacobian --input dvf.mha --output-img vf_jac.mha
        ((patient_id++))
    done
echo All done
