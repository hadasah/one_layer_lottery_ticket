from project.slurm_job import run_grid
import numpy as np
import os

SWEEP_NAME = "eval_1layer_wmt_base"
NUM_GPUS = 1
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
GIT_REPO_FOLDER = '/gscratch/zlab/margsli/gitfiles/one_layer_lottery_ticket'
TOP_LEVEL_EXPERIMENTS_FOLDER = f'{GIT_REPO_FOLDER}/experiments/'
TOP_LEVEL_DATA_FOLDER = f'{GIT_REPO_FOLDER}/data/data-bin'
DATA_FOLDER = f'{TOP_LEVEL_DATA_FOLDER}/wmt14_en_de_joined_dict'
OUTPUT_PATH = f'{TOP_LEVEL_EXPERIMENTS_FOLDER}/margsli/20220303/1layer_wmt_base_prune_0.5/1layer_wmt_base_prune_0.5'

cmd = (f'python {GIT_REPO_FOLDER}/scripts/average_checkpoints.py --inputs {OUTPUT_PATH} '
f'--num-epoch-checkpoints 10 --output {OUTPUT_PATH}/averaged_model.pt && '
f'python {GIT_REPO_FOLDER}/fairseq_cli/generate.py {DATA_FOLDER} --path {OUTPUT_PATH}/averaged_model.pt '
f' --beam 5 --remove-bpe')

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {},
        'named_args': {
        },
    },
}

for sweep_name, grid in grids.items():
    run_grid(
        grid,
        name_keys,
        sweep_name,
        user=os.environ['USER'],
        prefix=cmd,
        gpus=NUM_GPUS,
        cpus=5,
        nodes=1,
        account='bdata',
        partition='gpu-rtx6k',
        jobtime='6:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_sweep_name=True,
        top_level_experiments_folder=TOP_LEVEL_EXPERIMENTS_FOLDER,
        conda_env_name='cse517_project',
        dry_mode=DRY_MODE,
    )
