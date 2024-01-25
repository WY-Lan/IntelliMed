export CUDA_VISIBLE_DEVICES=5,6,7,8

export MODEL_BIOLINK=BioLinkBERT-base
export MODEL_BIOLINK_PATH=/data1/lanwy/lanwy/LinkBERT-main/LinkBERT/pretrain/BioLinkBERT-base

task=NCBI-disease_hf
datadir=/data1/lanwy/lanwy/LinkBERT-main/data/tokcls/$task
outdir=runs/$task/$MODEL_BIOLINK
mkdir -p $outdir
python3 -u train.py --model_name_or_path $MODEL_BIOLINK_PATH\
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test_all.json \
  --do_train --do_eval --do_predict \
  --fine_tune_type lora --\
   --per_device_train_batch_size 64 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 1e-5 --warmup_ratio 0.1 --num_train_epochs 1 --max_seq_length 512  \
  --save_strategy epoch --save_steps 20 --save_total_limit 1\
   --evaluation_strategy epoch --output_dir $outdir --overwrite_output_dir \
  --load_best_model_at_end --metric_for_best_model eval_overall_f1 --eval_steps 4\
  |& tee $outdir/log.txt &

wait