from project.slurm_job import run_grid
import numpy as np
import os

SWEEP_NAME = "1layer_iwslt_base_sweep"
NUM_GPUS = 1
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
GIT_REPO_FOLDER = '/gscratch/zlab/margsli/gitfiles/one_layer_lottery_ticket'
TOP_LEVEL_EXPERIMENTS_FOLDER = f'{GIT_REPO_FOLDER}/experiments/'
TOP_LEVEL_DATA_FOLDER = f'{GIT_REPO_FOLDER}/data/data-bin'
DATA_FOLDER = f'{TOP_LEVEL_DATA_FOLDER}/iwslt14.tokenized.de-en'

cmd = f'fairseq-train {DATA_FOLDER}'

grids = {
    SWEEP_NAME: {
        'fixed_args': '--seed 1 --fp16 --no-progress-bar \
            --max-epoch 55 --save-interval 1 --keep-last-epochs 5 \
            --optimizer adam --adam-betas \'(0.9, 0.98)\' --clip-norm 0.0 \
            --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
            --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
            --eval-bleu --eval-bleu-args \'{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}\' \
            --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --clip-norm 0.5 --mask-layernorm-type masked_layernorm \
            --mask-init standard --prune-method super_mask --mask-constant 0. \
            --scale-fan --share-decoder-input-output-embed',
        'positional_args': {},
        'named_args': {
            '--arch': ['masked_transformer_iwslt_de_en'],
            '--max-tokens': [4096],
            '--share-mask': ['layer_weights'],
            '--prune-ratio': [i for i in np.arange(0.1, 1.0, 0.1)],
            '--init': ['kaiming_uniform'],
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
        account='bdata',
        partition='gpu-2080ti',
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
