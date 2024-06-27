import papermill as pm
import os
from os.path import expanduser
import shutil

NB_INPUT_DIR = '{}/repos/HFLung-segmentation'.format(expanduser('~'))
NB_OUTPUT_DIR = '{}/papermill-output/compare-quantised-ctvi'.format(expanduser('~'))
NOTEBOOK_NAME = 'compare nnunet predictions with quantised PET and CTVI-Jac'

# make sure the notebook output directory exists
if os.path.exists(NB_OUTPUT_DIR):
    shutil.rmtree(NB_OUTPUT_DIR)
    print('deleted {}'.format(NB_OUTPUT_DIR))
os.makedirs(NB_OUTPUT_DIR)

DATASET_IDS = [160,161,162]
TEST_IDS = [0,1]
# CONFIG_SUFFIXES = ['','/2d']
CONFIG_SUFFIXES = ['/2d']

for config in CONFIG_SUFFIXES:
    for dataset_id in DATASET_IDS:
        for test_id in TEST_IDS:
            input_nb = '{}/{}.ipynb'.format(NB_INPUT_DIR, NOTEBOOK_NAME)
            output_nb = '{}{}/{} - dataset id {:02d} - test id {:02d}.ipynb'.format(NB_OUTPUT_DIR, config, NOTEBOOK_NAME, dataset_id, test_id)

            print('dataset id {:02d}, test id {:02d}: \'{}\''.format(dataset_id, test_id, output_nb))
            pm.execute_notebook(
                                input_path=input_nb,
                                output_path=output_nb,
                                parameters=dict(dataset_id=dataset_id, test_id=test_id, MODEL_CONFIG_SUFFIX=config)
                                )
            print('-------------\n')
