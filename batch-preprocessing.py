import papermill as pm
import glob
import os
from os.path import expanduser
import shutil

NB_INPUT_DIR = '{}/repos/CTVI-3D'.format(expanduser('~'))
NB_OUTPUT_DIR = '{}/papermill-output/pre-processing'.format(expanduser('~'))
NOTEBOOK_NAME = 'data pre-processing 3D plastimatch with post-DIR resampling'

# make sure the notebook output directory exists
if os.path.exists(NB_OUTPUT_DIR):
    shutil.rmtree(NB_OUTPUT_DIR)
    print('deleted {}'.format(NB_OUTPUT_DIR))
os.makedirs(NB_OUTPUT_DIR)

NUMBER_OF_PATIENTS = 20

for patient_id in range(1,NUMBER_OF_PATIENTS+1):

    input_nb = '{}/{}.ipynb'.format(NB_INPUT_DIR, NOTEBOOK_NAME)
    output_nb = '{}/{} - patient id {}.ipynb'.format(NB_OUTPUT_DIR, NOTEBOOK_NAME, patient_id)

    print('patient id {}: \'{}\' to \'{}\''.format(patient_id, input_nb, output_nb))
    pm.execute_notebook(
                        input_path=input_nb,
                        output_path=output_nb,
                        parameters=dict(patient_id=patient_id)
                        )
    print('-------------\n')
