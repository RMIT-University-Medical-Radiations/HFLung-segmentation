#!/bin/bash

patient_id=1
while [ $patient_id -le 20 ]
    do
        printf "%05d" $patient_id
        ((patient_id++))
    done
echo All done
