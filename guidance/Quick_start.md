## <span id="quik_start"> Quik evaluation of CoLMDriver

The steps for evaluating colmdriver on Interdrive benchmark

**Step 1:** Download checkpoints from [Google drive](https://drive.google.com/file/d/1z3poGdoomhujCNQtoQ80-BCO34GTOLb-/view?usp=sharing). We adopt InternVL2-4B for VLM and Qwen2.5-3B for LLM. The downloaded checkpoints of CoLMDriver should follow this structure:
```Shell
|--CoLMDriver
    |--ckpt
        |--colmdriver
            |--LLM
            |--perception
            |--VLM
            |--waypoints_planner
```

**Step 2:** Running VLM, LLM
```Shell
conda activate vllm
# VLM on call
CUDA_VISIBLE_DEVICES=6 vllm serve ckpt/colmdriver/VLM --port 1111 --max-model-len 8192 --trust-remote-code --enable-prefix-caching --gpu-memory-utilization 0.8

# LLM on call
CUDA_VISIBLE_DEVICES=7 vllm serve ckpt/colmdriver/LLM --port 8888 --max-model-len 4096 --trust-remote-code --enable-prefix-caching
```
Note: make sure that the selected ports (1111,8888) are not occupied by other services. If you use other ports, please modify values of key 'comm_client' and 'vlm_client' in `simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml` accordingly.

**Step 3:** Run CARLA, run CoLMDriver
```Shell
conda activate colmdriver

# Start CARLA server, if port 2000 is already in use, choose another
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=2000 -prefer-nvidia

# Open another terminal

# Evaluate CoLMDriver on Interdrive(92 tests), 0 is CUDA id, 2000 is CARLA port (be consistent with your CARLA server)
bash scripts/eval/eval_mode.sh 0 2000 colmdriver ideal Interdrive_all

# Evaluate CoLMDriver on Interdrive, considering inference latency
bash scripts/eval/eval_mode.sh 0 2000 colmdriver realtime Interdrive_all

# Evaluate CoLMDriver on Interdrive(46 tests), in scenarios with no NPC, only collaborative vehicles
bash scripts/eval/eval_mode.sh 0 2000 colmdriver ideal Interdrive_no_npc

# Evaluate CoLMDriver on Interdrive(46 tests), in scenarios with NPC (other traffic participants)
bash scripts/eval/eval_mode.sh 0 2000 colmdriver ideal Interdrive_npc
```

The results will be saved at `results/results_driving_colmdriver`, to summarize the results, use:
```Shell
python visualization/result_analysis.py results/results_driving_colmdriver
```

It's recommended to run LLM/VLM/CARLA_server/CoLMDriver_evaluation in 4 distinct terminals.

## <span id="quik_start_others"> Evaluation of baselines
Setup and get ckpts.

| Methods   | TCP | CoDriving               |
|-----------|---------|---------------------------|
| Installation Guide  | [github](https://github.com/OpenDriveLab/TCP)  | [github](https://github.com/CollaborativePerception/V2Xverse) |
| Checkpoints     |  [google drive](https://drive.google.com/file/d/1D-10aMUAOPk1yiOr-PvSOJMS_xi_eR7U/view?usp=sharing)  |  [google drive](https://drive.google.com/file/d/1Izg9wZ3ktR-mwn7J_ZqxrwBmtI1YJ6Xi/view?usp=sharing)   |

The downloaded checkpoints should follow this structure:
```Shell
|--CoLMDriver
    |--ckpt
        |--codriving
            |--perception
            |--planning
        |--TCP
            |--new.ckpt
```

Evaluate TCP, CoDriving on Interdrive benchmark:

```Shell
# Start CARLA server, if port 2000 is already in use, choose another
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=2000 -prefer-nvidia

# Evaluate TCP on Interdrive
bash scripts/eval/eval_mode.sh 0 2000 tcp ideal Interdrive_all

# Evaluate CoDriving on Interdrive
bash scripts/eval/eval_mode.sh 0 2000 codriving ideal Interdrive_all
```

## <span id="Shutdown"> Shut down simulation on Linux
CARLA processes may fail to stop，please kill them in time.

Display your processes
~~~
ps U usrname | grep PROCESS_NAME(eg. python，carla)
~~~
Kill process
~~~
kill -9 PID
~~~
Kill all carla-related processes
~~~
ps -def |grep 'carla' |cut -c 9-15| xargs kill -9
pkill -u username -f carla
~~~