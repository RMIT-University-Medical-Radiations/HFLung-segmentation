import os
import nibabel as nib
import numpy as np
import shutil
import glob
import random
import json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

preprocessed_dir = '/mnt/data/datasets/RNSH_HFlung/pre-processed-plastimatch/stack'
nnUNet_raw='/mnt/data/datasets/RNSH_HFlung/nnU-Net-processing/nnUNet_raw'
base_task_id = 150
task_name = "RNSH_HFlung"

def choose_test_patients(patient_ids, number_of_test_patients, number_of_test_sets):
    result = []
    while len(result) < number_of_test_sets:
        s = random.sample(list(patient_ids), number_of_test_patients)
        if (s not in result) and (s[::-1] not in result):
            result.append(s)
        else:
            print('{} is already present'.format(s))
    return result

patient_test_sets = choose_test_patients(patient_ids=np.arange(1,20+1), number_of_test_patients=2, number_of_test_sets=5)
print(patient_test_sets)
for test_set_idx,test_set in enumerate(patient_test_sets):
    foldername = 'Dataset{:03d}_{}'.format(base_task_id+test_set_idx, task_name)
    print('test set {}: {}'.format(test_set, foldername))

    # setting up nnU-Net folders
    out_base_dir = '{}/{}'.format(nnUNet_raw, foldername)
    training_image_dir = '{}/{}'.format(out_base_dir, "imagesTr")
    training_label_dir = '{}/{}'.format(out_base_dir, "labelsTr")
    test_image_dir = '{}/{}'.format(out_base_dir, "imagesTs")
    test_label_dir = '{}/{}'.format(out_base_dir, "labelsTs")

    if os.path.exists(training_image_dir):
        shutil.rmtree(training_image_dir)
    if os.path.exists(training_label_dir):
        shutil.rmtree(training_label_dir)

    if os.path.exists(test_image_dir):
        shutil.rmtree(test_image_dir)
    if os.path.exists(test_label_dir):
        shutil.rmtree(test_label_dir)

    os.makedirs(training_image_dir)
    os.makedirs(training_label_dir)
    os.makedirs(test_image_dir)
    os.makedirs(test_label_dir)

    path = '{}/Patient*.npy'.format(data_dir)
    file_l = sorted(glob.glob(path))
    test_file_l = [ file_l[i-1] for i in test_set ]
    training_file_l = list(set(file_l) - set(test_file_l))

    patient_map_d = {'training':[], 'test':[]}

    print('processing training set')
    for c,f in enumerate(training_file_l):
        print(f)
        # split the patient file contents into channels
        patient_arr = np.load(f)
        patient_arr = np.moveaxis(patient_arr, 1, -1)  # move the z-axis to be last
        
        exh_img = nib.Nifti1Image(patient_arr[0], np.eye(4))  # identity matrix for transform
        inh_img = nib.Nifti1Image(patient_arr[1], np.eye(4))
        label_img = nib.Nifti1Image(patient_arr[4], np.eye(4))
        
        case_id_str = '{}_{:04d}'.format(task_name, c)
        
        # 2-channel inputs
        exh_img.to_filename('{}/{}_{:04d}.nii.gz'.format(training_image_dir, case_id_str, 0))  # channel 0
        inh_img.to_filename('{}/{}_{:04d}.nii.gz'.format(training_image_dir, case_id_str, 1))  # channel 1
        
        # 1-channel labels
        label_img.to_filename('{}/{}.nii.gz'.format(training_label_dir, case_id_str))

        patient_map_d['training'].append([f,case_id_str])

    print('processing test set')
    for c,f in enumerate(test_file_l):
        print(f)
        # split the patient file contents into channels
        patient_arr = np.load(f)
        patient_arr = np.moveaxis(patient_arr, 1, -1)  # move the z-axis to be last
        
        exh_img = nib.Nifti1Image(patient_arr[0], np.eye(4))  # identity matrix for transform
        inh_img = nib.Nifti1Image(patient_arr[1], np.eye(4))
        label_img = nib.Nifti1Image(patient_arr[4], np.eye(4))
        
        case_id_str = '{}_{:04d}'.format(task_name, c)
        
        # 2-channel inputs
        exh_img.to_filename('{}/{}_{:04d}.nii.gz'.format(test_image_dir, case_id_str, 0))  # channel 0
        inh_img.to_filename('{}/{}_{:04d}.nii.gz'.format(test_image_dir, case_id_str, 1))  # channel 1
        
        # 1-channel labels
        label_img.to_filename('{}/{}.nii.gz'.format(test_label_dir, case_id_str))

        patient_map_d['test'].append([f,case_id_str])

    # write the dataset file
    generate_dataset_json(out_base_dir,
                        channel_names={0: 'exh_ct', 1: 'inh_ct'},
                        labels={
                            'background': 0,
                            'high function': 1,
                            'medium function': 2,
                            'low function': 3
                        },
                        num_training_cases=len(training_file_l),
                        file_ending='.nii.gz',
                        regions_class_order=(1, 2, 3),
                        license='',
                        reference='',
                        dataset_release='1.0')

    # write the patient ID mapping
    with open('{}/patient-mapping-{}.json'.format(nnUNet_raw, foldername), 'w') as fp:
        json.dump(patient_map_d, fp)
