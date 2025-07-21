# CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving (ICCV2025)

[Paper link](https://arxiv.org/pdf/2503.08683)

This repository contains the official PyTorch implementation of paper "CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving".

![framework](simulation/demo/framework.jpeg)

## <span id="Todo"> Todo
- [x] Checkpoints release of CoLMDriver
- [ ] Training of CoLMDriver
  - [x] perception
  - [x] planning
  - [ ] MLLM
- [ ] Interdrive evaluation
  - [x] CoLMDriver
  - [x] CoDriving
  - [x] TCP
  - [ ] LMDrive
  - [ ] UniAD
  - [ ] VAD

## Contents
1. [Installation](guidance/Installation.md)
2. [Quik evaluation](guidance/Quick_start.md)
3. [Dataset](guidance/Dataset.md)
4. [Training](guidance/Training.md)


## <span id="Acknowledgements"> Acknowledgements
This implementation is based on code from several repositories.
- [V2Xverse](https://github.com/CollaborativePerception/V2Xverse)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)

## Citation
```
@article{liu2025colmdriver,
  title={CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving},
  author={Liu, Changxing and Liu, Genjia and Wang, Zijun and Yang, Jinchang and Chen, Siheng},
  journal={arXiv preprint arXiv:2503.08683},
  year={2025}
}
```