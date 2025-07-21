## <span id="dataset"> Dataset
The dataset for training CoLMDriver is obtained from [V2Xverse](https://github.com/CollaborativePerception/V2Xverse), which contains experts behaviors in CARLA. You may get the dataset in two ways:
- Download from [this huggingface repository](https://huggingface.co/datasets/gjliu/V2Xverse).
- Generate the dataset by yourself, following this [guidance](https://github.com/CollaborativePerception/V2Xverse).

The dataset should be linked/stored under `external_paths/data_root/` follow this structure:
```Shell
|--data_root
    |--weather-0
        |--data
            |--routes_town{town_id}_{route_id}_w{weather_id}_{datetime}
                |--ego_vehicle_{vehicle_id}
                    |--2d_bbs_{direction}
                    |--3d_bbs
                    |--actors_data
                    |--affordances
                    |--bev_visibility
                    |--birdview
                    |--depth_{direction}
                    |--env_actors_data
                    |--lidar
                    |--lidar_semantic_front
                    |--measurements
                    |--rgb_{direction}
                    |--seg_{direction}
                    |--topdown
                |--rsu_{vehicle_id}
                |--log
            ...
```