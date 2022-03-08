from project.slurm_job import run_grid
import numpy as np
import os

SWEEP_NAME = "eval_1layer_iwslt_base_sweep_embed"
NUM_GPUS = 1
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
GIT_REPO_FOLDER = '/gscratch/zlab/margsli/gitfiles/one_layer_lottery_ticket'
TOP_LEVEL_EXPERIMENTS_FOLDER = f'{GIT_REPO_FOLDER}/experiments/'
TOP_LEVEL_DATA_FOLDER = f'{GIT_REPO_FOLDER}/data/data-bin'
DATA_FOLDER = f'{TOP_LEVEL_DATA_FOLDER}/iwslt14.tokenized.de-en'
# ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ratios = [0.6, 0.7, 0.8]
cmd = f'python /mmfs1/gscratch/zlab/margsli/gitfiles/one_layer_lottery_ticket/fairseq_cli/generate.py {DATA_FOLDER}'
# OUTPUT_PATHS = [f'{TOP_LEVEL_EXPERIMENTS_FOLDER}/margsli/20220227/1layer_iwslt_base_sweep_embed/1layer_iwslt_base_sweep_embed_prune-ratio={r}/checkpoint_best.pt' for r in ratios]
OUTPUT_PATHS = [f'{TOP_LEVEL_EXPERIMENTS_FOLDER}/margsli/20220305/1layer_iwslt_base_sweep_embed/1layer_iwslt_base_sweep_embed_prune-ratio={r}/checkpoint_best.pt' for r in ratios]

grids = {
    SWEEP_NAME: {
        'fixed_args': '--batch-size 128  --beam 5 \
                    --lenpen 1.0 --remove-bpe --log-format simple --source-lang de --target-lang en',
        'positional_args': {},
        'named_args': {
            '--path': OUTPUT_PATHS,
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
        output_dir_name='--save-dir',
        conda_env_name='cse517_project',
        dry_mode=DRY_MODE,
    )
