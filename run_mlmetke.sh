#!/bin/bash
TOTAL_UPDATES=125000     # Total number of training steps
WARMUP_UPDATES=10000     # Warmup the learning rate over this many updates
LR=6e-04                 # Peak LR for polynomial LR scheduler
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size # Set to 2 in DARG 
NUM_NODES=1		# Number of machines
ROBERTA_PATH="/projects/KEPLER/checkpoints_roberta_mlm_125k/checkpoint_best.pt" # Path to the original roberta model
CHECKPOINT_PATH="checkpoints_mlmke_nheads_disabled_125k" #Directory to store the checkpoints
UPDATE_FREQ=`expr 784 / $NUM_NODES` # Increase the batch size
DATA_DIR="/projects/KEPLER_DATA"
MLM_DATA="/projects/KEPLER_DATA/CITE_MLM/data-bin/CITE"

#Path to the preprocessed KE dataset, each item corresponds to a data directory for one epoch
#KE_DATA=$DATA_DIR/CITE_KE/CITE1_0:$DATA_DIR/CITE_KE/CITE1_1:$DATA_DIR/CITE_KE/CITE1_2:$DATA_DIR/CITE_KE/CITE1_3
KE_DATA=$DATA_DIR/CITE_KE/CITE1_0

DIST_SIZE=`expr $NUM_NODES \* 1`

fairseq-train $MLM_DATA \
        --KEdata $KE_DATA \
        --restore-file $ROBERTA_PATH \
        --save-dir $CHECKPOINT_PATH \
        --max-sentences $MAX_SENTENCES \
        --tokens-per-sample 512 \
        --task MLMetKE \
        --sample-break-mode complete \
        --required-batch-size-multiple 1 \
        --arch roberta_base \
        --criterion MLMetKE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --negative-sample-size 1 \
        --ke-model TransE \
        --init-token 0 \
        --separator-token 2 \
        --gamma 4 \
        --nrelation 1 \
        --skip-invalid-size-inputs-valid-test \
        --fp16 --fp16-init-scale 2 --threshold-loss-scale 1 --fp16-scale-window 128 \
        --reset-optimizer --distributed-world-size ${DIST_SIZE} --ddp-backend no_c10d --distributed-port 23456 \
        --log-format simple --log-interval 1 \
        --no-epoch-checkpoints
