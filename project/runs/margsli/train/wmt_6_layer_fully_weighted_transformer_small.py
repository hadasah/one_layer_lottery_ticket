from project.slurm_job import run_grid
import numpy as np
import os

SWEEP_NAME = "6layer_wmt_fully_weighted"
NUM_GPUS = 8
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
GIT_REPO_FOLDER = '/gscratch/zlab/margsli/gitfiles/one_layer_lottery_ticket'
TOP_LEVEL_EXPERIMENTS_FOLDER = f'{GIT_REPO_FOLDER}/experiments/'
TOP_LEVEL_DATA_FOLDER = f'{GIT_REPO_FOLDER}/data/data-bin'
DATA_FOLDER = f'{TOP_LEVEL_DATA_FOLDER}/wmt14_en_de_joined_dict'

cmd = f'fairseq-train {DATA_FOLDER}'

grids = {
    SWEEP_NAME: {
        'fixed_args': '--seed 1 --dropout 0.2 --no-progress-bar --fp16 \
            --share-decoder-input-output-embed --optimizer adam --adam-betas \'(0.9, 0.98)\' --clip-norm 0.0 \
            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 \
            --lr 1e-3 --update-freq 1 --log-interval 50 \
            --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
            --max-tokens 4096 --distributed-world-size 8 --distributed-port 61024 \
            --ddp-backend no_c10d --keep-interval-updates 20 --keep-last-epochs 10 --max-epoch 100 \
            --share-decoder-input-output-embed',
        'positional_args': {},
        'named_args': {
            '--arch': ['transformer_wmt_en_de'],
            '--max-tokens': [4096],
            '--wandb-project': ['cse517-project'],
            '--wandb-entity': ['cse517-project-wi22'],
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
        account='zlab',
        partition='gpu-rtx6k',
        jobtime='12:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_sweep_name=True,
        top_level_experiments_folder=TOP_LEVEL_EXPERIMENTS_FOLDER,
        output_dir_name='--save-dir',
        conda_env_name='cse517_project',
        dry_mode=DRY_MODE,
    )
