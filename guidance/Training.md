## <span id="train"> Training

Training of three parts: intention-guided waypoints, vlm, perception.

### Perception module
Our perception module follows [CoDriving](https://github.com/CollaborativePerception/V2Xverse).
To train perception module from scratch or a continued checkpoint, run the following commonds:
```Shell
# Single GPU training
python opencood/tools/train.py -y opencood/hypes_yaml/v2xverse/colmdriver_multiclass_config.yaml [--model_dir ${CHECKPOINT_FOLDER}]

# DDP training
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y opencood/hypes_yaml/v2xverse/fcooper_multiclass_config.yaml [--model_dir ${CHECKPOINT_FOLDER}]

# Offline testing of perception
python opencood/tools/inference_multiclass.py --model_dir ${CHECKPOINT_FOLDER}
```
The training outputs can be found at `opencood/logs`.
Arguments Explanation:
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune or continue-training. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder. In this case, ${CONFIG_FILE} can be `None`,
- `--nproc_per_node` indicate the GPU number you will use.

### Planning module
Given a checkpoint of perception module, we freeze its parameters and train the down-stream planning module in an end-to-end paradigm. The planner gets BEV perception feature and occupancy map as input and targets to predict the future waypoints of ego vehicle.

Train the planning module with a given perception checkpoint on multiple GPUs:
```Shell
# Train planner
bash scripts/train/train_planner_e2e.sh $GPU_ids $num_GPUs $perception_ckpt $planner_config $planner_ckpt_resume $name_of_log $save_path

# Example
bash scripts/train/train_planner_e2e.sh 0,1 2 ckpt/colmdriver/percpetion covlm_cmd_extend_adaptive_20 None log ./ckpt/colmdriver_planner

# Offline test
bash scripts/eval/eval_planner_e2e.sh 0,1 ckpt/colmdriver/percpetion covlm_cmd_extend_adaptive_20 ckpt/colmdriver/waypoints_planner/epoch_26.ckpt ./ckpt/colmdriver_planner
```

### Tuning MLLM
Coming soon.