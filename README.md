# Todo List
- [ ] fix(eval_policy): Fix the logic for destroying the video pipeline

---
# Data Structure！
```
episode1.hdf5
├── endpose                          (float64, shape: [147, 14])
├── joint_action/
│   ├── left_arm                    (float64, shape: [147, 6])
│   ├── left_gripper               (float64, shape: [147])
│   ├── right_arm                  (float64, shape: [147, 6])
│   ├── right_gripper              (float64, shape: [147])
│   └── vector                     (float64, shape: [147, 14])
├── observation/
│   ├── front_camera/
│   │   ├── cam2world_gl          (float32, shape: [147, 4, 4])
│   │   ├── depth                 (float64, shape: [147, 240, 320])
│   │   ├── extrinsic_cv         (float32, shape: [147, 3, 4])
│   │   ├── intrinsic_cv         (float32, shape: [147, 3, 3])
│   │   └── rgb                   (|S17330, shape: [147])
│   ├── head_camera/
│   │   ├── cam2world_gl          (float32, shape: [147, 4, 4])
│   │   ├── depth                 (float64, shape: [147, 240, 320])
│   │   ├── extrinsic_cv         (float32, shape: [147, 3, 4])
│   │   ├── intrinsic_cv         (float32, shape: [147, 3, 3])
│   │   └── rgb                   (|S19188, shape: [147])
│   ├── left_camera/
│   │   ├── cam2world_gl          (float32, shape: [147, 4, 4])
│   │   ├── depth                 (float64, shape: [147, 240, 320])
│   │   ├── extrinsic_cv         (float32, shape: [147, 3, 4])
│   │   ├── intrinsic_cv         (float32, shape: [147, 3, 3])
│   │   └── rgb                   (|S17970, shape: [147])
│   └── right_camera/
│       ├── cam2world_gl          (float32, shape: [147, 4, 4])
│       ├── depth                 (float64, shape: [147, 240, 320])
│       ├── extrinsic_cv         (float32, shape: [147, 3, 4])
│       ├── intrinsic_cv         (float32, shape: [147, 3, 3])
│       └── rgb                   (|S5962, shape: [147])
└── pointcloud                      (float32, shape: [147, 1024, 6])

```
---
# ⚠️ Important Notes When Using the L40 Cluster

## 1. ✅ Vulkan Setup

* Please verify that the command `vulkaninfo` can correctly list **all four L40 GPUs**.

* If `vulkaninfo` fails to display the GPUs, refer to this GitHub issue for troubleshooting:
  👉 [NVIDIA Container Toolkit Issue #16](https://github.com/NVIDIA/nvidia-container-toolkit/issues/16)

* The Vulkan **ICD (Installable Client Driver)** configuration file should look like this:

  ```json
  {
    "file_format_version": "1.0.0",
    "ICD": {
      "library_path": "libGLX_nvidia.so.0",
      "api_version": "1.3.277"
    }
  }
  ```

## 2. 🎥 Installing FFmpeg in Docker

Docker images typically **do not include `ffmpeg`** by default. To enable video/audio processing features, follow these steps:

### Step-by-Step:

1. **Install `libvpx7` first**
   Because our cluster uses the `amd64` CPU architecture, you must manually install `libvpx7` before installing `ffmpeg`.

   A pre-downloaded `.deb` package is located at:

   ```
   /data/sea_disk0/cuihz/code/libvpx7_1.12.0-1+deb12u4_amd64.deb
   ```

   Install it with:

   ```bash
   sudo apt install /data/sea_disk0/cuihz/code/libvpx7_1.12.0-1+deb12u4_amd64.deb
   ```

2. **Then install FFmpeg**:

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

## 3. 🔧 Required for Bash Script Compatibility: `gettext`

Some of our bash scripts rely on commands such as `envsubst`, which are **not available in minimal Docker images** by default. These tools are provided by the `gettext` package.

To ensure full compatibility, run:

```bash
sudo apt install gettext
```

Without this step, some scripts may fail with
---
# 🛠️ Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for installation instructions. It takes about 20 minutes for installation.

# 🤷‍♂️ Tasks Informations
See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

# 🧑🏻‍💻 Usage 

> Please Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for more details.

## Data Collection
We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
However, we strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

## 1. Task Running and Data Collection
Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

## 2. Task Config
See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

# 🚴‍♂️ Policy Baselines
## Policies Support
[DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)

[TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA

# 🏄‍♂️ Experiment & LeaderBoard

> We recommend that the RoboTwin Platform can be used to explore the following topics: 
> 1. single - task fine - tuning capability
> 2. visual robustness
> 3. language diversity robustness (language condition)
> 4. multi-tasks capability
> 5. cross-embodiment performance

Coming Soon.

# 👍 Citations
If you find our work useful, please consider citing:

<b>RoboTwin 2.0</b>: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
```
Coming Soon.
```

<b>RoboTwin</b>: Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>
```
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

<b>RoboTwin</b>: Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

# 😺 Acknowledgement

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

Contact [Tianxing Chen](https://tianxingchen.github.io) if you have any questions or suggestions.

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.
