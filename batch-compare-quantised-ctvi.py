import papermill as pm
import os
from os.path import expanduser
import shutil

NB_INPUT_DIR = '{}/repos/HFLung-segmentation'.format(expanduser('~'))
NB_OUTPUT_DIR = '{}/papermill-output/compare-quantised-ctvi'.format(expanduser('~'))
NOTEBOOK_NAME = 'compare nnunet predictions with quantised PET and CTVI-Jac'

DATASET_IDS = [160,161,162]
TEST_IDS = [0,1]
CONFIG_SUFFIXES = ['','/2d']

for config in CONFIG_SUFFIXES:
    output_nb_path = '{}{}'.format(NB_OUTPUT_DIR, config.replace('/','-'))
    # make sure the notebook output directory exists
    if os.path.exists(output_nb_path):
        shutil.rmtree(output_nb_path)
        print('deleted {}'.format(output_nb_path))
    os.makedirs(output_nb_path)
    for dataset_id in DATASET_IDS:
        for test_id in TEST_IDS:
            input_nb = '{}/{}.ipynb'.format(NB_INPUT_DIR, NOTEBOOK_NAME)
            output_nb = '{}/{} - dataset id {:02d} - test id {:02d}.ipynb'.format(output_nb_path, NOTEBOOK_NAME, dataset_id, test_id)

            if not os.path.exists(output_nb_path):
                os.makedirs(output_nb_path)

            print('dataset id {:03d}, test id {:02d}: \'{}\''.format(dataset_id, test_id, output_nb))
            pm.execute_notebook(
                                input_path=input_nb,
                                output_path=output_nb,
                                parameters=dict(DATASET_ID=dataset_id, TEST_ID=test_id, MODEL_CONFIG_SUFFIX=config)
                                )
            print('-------------\n')
