import papermill as pm
import os
from os.path import expanduser
import shutil

NB_INPUT_DIR = '{}/repos/HFLung-segmentation'.format(expanduser('~'))
NB_OUTPUT_DIR = '{}/papermill-output/compare-unquantised-ctvi'.format(expanduser('~'))
NOTEBOOK_NAME = 'compare unquantised CTVI with PET'

# make sure the notebook output directory exists
if os.path.exists(NB_OUTPUT_DIR):
    shutil.rmtree(NB_OUTPUT_DIR)
    print('deleted {}'.format(NB_OUTPUT_DIR))
os.makedirs(NB_OUTPUT_DIR)

NUMBER_OF_PATIENTS = 20

for patient_id in range(1,NUMBER_OF_PATIENTS+1):

    input_nb = '{}/{}.ipynb'.format(NB_INPUT_DIR, NOTEBOOK_NAME)
    output_nb = '{}/{} - patient id {:02d}.ipynb'.format(NB_OUTPUT_DIR, NOTEBOOK_NAME, patient_id)

    print('patient id {:02d}: \'{}\' to \'{}\''.format(patient_id, input_nb, output_nb))
    pm.execute_notebook(
                        input_path=input_nb,
                        output_path=output_nb,
                        parameters=dict(patient_id=patient_id)
                        )
    print('-------------\n')
