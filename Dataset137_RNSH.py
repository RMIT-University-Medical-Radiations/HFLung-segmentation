from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
import nibabel as nib
import numpy as np
import shutil
import glob


if __name__ == '__main__':
    data_dir = '/home/daryl/datasets/RNSH_HFlung/training-set'

    task_id = 138
    task_name = "RNSH_HFlung"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base_dir = '{}/{}'.format(nnUNet_raw, foldername)
    image_dir = '{}/{}'.format(out_base_dir, "imagesTr")
    label_dir = '{}/{}'.format(out_base_dir, "labelsTr")
    maybe_mkdir_p(image_dir)
    maybe_mkdir_p(label_dir)

    path = '{}/Patient*.npy'.format(data_dir)
    file_l = sorted(glob.glob(path))


    for c,f in enumerate(file_l):
        # split the patient file contents into channels
        patient_arr = np.load(f)

        exh_img = nib.Nifti1Image(patient_arr[0], np.eye(4))  # identity matrix for transform
        inh_img = nib.Nifti1Image(patient_arr[1], np.eye(4))
        label_img = nib.Nifti1Image(patient_arr[3], np.eye(4))

        case_id_str = '{}_{:04d}'.format(task_name, c)

        # 2-channel inputs
        exh_img.to_filename('{}/{}_{:04d}.nii.gz'.format(image_dir, case_id_str, 0))  # channel 0
        inh_img.to_filename('{}/{}_{:04d}.nii.gz'.format(image_dir, case_id_str, 1))  # channel 1

        # 1-channel labels
        label_img.to_filename('{}/{}.nii.gz'.format(label_dir, case_id_str))

    generate_dataset_json(out_base_dir,
                          channel_names={0: 'exh_ct', 1: 'inh_ct'},
                          labels={
                              'background': 0,
                              'high function': 1,
                              'medium function': 2,
                              'low function': 3
                          },
                          num_training_cases=len(file_l),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='',
                          reference='',
                          dataset_release='1.0')
