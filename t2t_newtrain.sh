USR_DIR=./
PROBLEM=translate_ende_custom
MODEL=transformer
DATA_DIR=./data_dir
TMP_DIR=./tmp_dir
TRAIN_DIR=./training
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# specify file to be translated here
#DECODE_FILE=

# specify output translation path
#OUTPUT_FILE=

t2t-trainer \
  --data_dir=$DATA_DIR \
  --t2t_usr_dir=$USR_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=$TRAIN_DIR

# comment --decode_interactive and uncomment --decode_to_file/--decode_from_file to enable file translation
BEAN_SIZE=4
ALPHA=0.6
t2t-decoder \
  --data_dir=$DATA_DIR\
  --t2t_usr_dir=$USR_DIR\
  --problem=$PROBLEM\
  --model=$MODEL\
  --hparams_set=transformer_base_single_gpu\
  --output_dir=$TRAIN_DIR \
#  --decode_from_file=$DECODE_FILE\
#  --decode_to_file=$OUTPUT_FILE\
  --decode_interactive
