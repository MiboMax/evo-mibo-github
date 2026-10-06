# Evo-Mibo

Evo-Mibo is an HSR-oriented research fork of Evo-1 for vision-language-action
mobile manipulation. It keeps the Evo-1 design philosophy of using a large VLM
backbone with a flow-matching action expert, and extends it for tasks where the
robot must reason about mobile-base motion, object-target geometry, and
depth-based spatial cues.

This repository is prepared as paper/demo code. It contains the core training,
dataset, and model implementation, but does not include pretrained weights,
checkpoints, raw datasets, logs, rollout videos, or evaluation-only scripts.

Although the example scripts in this codebase were developed around HSR tomato
can manipulation, the implementation is task-agnostic. Any task can be trained
if it is converted to the expected LeRobot-style format with RGB images, robot
state, actions, language instruction, and optional scene-slot supervision.

## Main Differences from Evo-1

Compared with the original Evo-1 arm-manipulation setup, Evo-Mibo introduces the
following changes:

1. **HSR mobile-manipulation state/action interface.** The code supports HSR
   robot state vectors such as 15D robot-only state and extended 27D/43D state
   variants with spatial slot information. The action head predicts action
   chunks with horizon `H=50`.

2. **Inverse-depth visual input.** In addition to head RGB and hand RGB,
   Evo-Mibo can use a third visual stream: hand-mounted inverse depth. The depth
   image is encoded as a visual input rather than being appended directly to the
   low-dimensional robot state.

3. **Slot-based 3D spatial memory.** Evo-Mibo adds compact scene slots that
   represent object/target spatial structure around the robot. These slots are
   encoded into memory tokens and fused before the flow-matching action expert.

4. **End-to-end slot prediction option.** In simulation, ground-truth slots can
   supervise a visual slot predictor. At deployment time, the action model can
   run from images and robot state without requiring simulator ground-truth
   object positions.

5. **Dataset conversion utilities.** The repository includes conversion scripts
   for RoboManipBaselines/HSR demonstrations into the LeRobot-style format used
   by training.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `config.py` | Main model/config definitions, including state dimension, action dimension, image-view settings, and action horizon. |
| `scripts/Evo1.py` | High-level Evo-1/Evo-Mibo model wrapper and forward path into the VLM and action head. |
| `scripts/train.py` | Main training entrypoint. Handles dataset loading, slot targets, stage-wise freezing, and loss logging. |
| `model/action_head/flow_matching.py` | Flow-matching action head, visual slot predictor, and geometric memory state encoder. |
| `dataset/lerobot_dataset_pretrain_mp.py` | Dataset preprocessing before samples are passed to the model. Handles image views, robot state, actions, and scene slots. |
| `scripts/convert_rmb_hsr_to_lerobot.py` | Converts raw HSR/RoboManipBaselines demonstrations to the training format, including optional inverse depth and slot fields. |
| `scripts/Evo1_server.py` | WebSocket inference server used by simulation rollout. |
| `scripts/rollout_hsr_tomato_can_box.py` | Demo MuJoCo rollout client for the HSR tomato-can-box task. |
| `model/initial_scene_slots.py` | Lightweight perceptor used to estimate initial scene slots from reset-time RGB/depth observations. |
| `scripts/train_initial_scene_slot_perceptor.py` | Standalone training script for the initial scene slot perceptor. |
| `scripts/fill_lerobot_dataset_predicted_slots.py` | Utility for filling a dataset with predicted slots from the perceptor. |
| `dataset/*.yaml` | Example dataset configs. Update paths and dimensions for new tasks. |
| `ds_config/` | DeepSpeed configuration files. |
| `docs/` | Additional notes on the mibo4 slot-memory design. |

## Inverse Depth Architecture

The inverse-depth extension is implemented as a third image stream. Metric depth
from the hand camera is converted into an inverse-depth image where closer
points receive larger values. This preserves useful object-distance structure
while letting the existing vision backbone process the signal as an image.

Key implementation locations:

- Conversion and encoding:
  `scripts/convert_rmb_hsr_to_lerobot.py`
  - `--include-hand-depth`
  - `--hand-depth-encoding inverse`
  - depth normalization and inverse-depth video export

- Dataset view mapping:
  `dataset/*.yaml`
  - `view_map.image_1`: typically head RGB
  - `view_map.image_2`: typically hand RGB
  - `view_map.image_3`: typically hand inverse depth

- Dataset preprocessing:
  `dataset/lerobot_dataset_pretrain_mp.py`
  - reads `view_map`
  - loads all configured image views
  - returns the multi-view image batch consumed by the model

No special depth-specific VLM backbone is required. Inverse depth is treated as
an additional visual view and encoded by the same visual-token pipeline.

## Slot-Based 3D Spatial Memory

The slot-memory module is designed to give the action expert a compact geometric
description of the scene. In the compact setting, each slot stores 2D relative
position and a validity mask, for example:

```text
slot_k = [dx_k, dy_k, valid_k]
```

Multiple slots can represent task-relevant objects, targets, or distractors.
Invalid slots are masked out. Self-relations are excluded when computing
slot-slot geometry.

The main components are implemented in:

- `model/action_head/flow_matching.py`
  - `VisualSceneSlotPredictor`: predicts compact slots from visual/VLM tokens.
  - `GeometricMemoryStateEncoder`: converts robot state and scene slots into
    geometry-aware memory tokens.
  - Flow-matching action head: fuses VLM tokens, state tokens, and memory tokens
    before predicting action chunks.

- `scripts/train.py`
  - reads `scene_slots` and `scene_slots_mask` from the dataset
  - enables predicted-slot training with `--predict_scene_slots`
  - adds auxiliary slot loss with `--slot_aux_loss_weight`

- `dataset/lerobot_dataset_pretrain_mp.py`
  - extracts slot tensors from each demonstration sample
  - provides compact slots and masks to the model

During simulation training, slot supervision can come from known object poses.
During real deployment, the model can instead rely on the learned visual slot
predictor or on an external perceptor that estimates approximate slots from
camera observations.

The training objective with slot prediction is:

```text
L_total = L_action + lambda_slot * L_slot
```

where `L_action` is the flow-matching action loss and `L_slot` supervises the
predicted compact scene slots.

## Training Workflow

### Environment

The training code and the MuJoCo rollout code can share one environment. A
typical setup is:

```bash
conda create -n evo_mibo python=3.10 -y
conda activate evo_mibo

pip install --upgrade pip
pip install -r requirements.txt
```

For rollout, install RoboManipBaselines as an editable package because the HSR
MuJoCo environment and assets are registered there:

```bash
git clone <RoboManipBaselines-repo-url> ../RoboManipBaselines
pip install -e ../RoboManipBaselines
```

For headless machines, MuJoCo usually works with:

```bash
export MUJOCO_GL=egl
```

If `deepspeed` or `flash-attn` fails to build, install PyTorch and CUDA-matched
build tools first, then reinstall the failed package.

### Demo Dataset

The demo dataset used by the example scripts is:

```text
HSR_TomatoCanBoxRandomRectRealMatchMix80_RGB15_HandInvDepth005to200_State27_BinaryGripper025_InitialScene_20260621_20260720_lerobot_v21
```

It contains HSR tomato-can-box demonstrations with:

- head RGB
- hand RGB
- hand inverse depth
- 15D robot state
- compact 2D scene slots, giving 27D state total
- 9D action
- language instruction

The public demo dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/Maxmibo/HSR-TomatoCanBoxRandomRectRealMatchMix80
```

```bash
export HF_DATASET_REPO="Maxmibo/HSR-TomatoCanBoxRandomRectRealMatchMix80"
mkdir -p data
hf download "${HF_DATASET_REPO}" \
  --repo-type dataset \
  --local-dir data/HSR_TomatoCanBoxRandomRectRealMatchMix80_RGB15_HandInvDepth005to200_State27_BinaryGripper025_InitialScene_20260621_20260720_lerobot_v21
```

The matching dataset config is:

```text
dataset/config_demo_hsr_tomato_randomrect_realmatch_mix80_state27.yaml
```

If you place the dataset somewhere else, edit the `path:` field in that YAML or
pass another config through `DATASET_CONFIG=...`.

### Demo Training

The recommended mibo4 workflow is:

1. Convert raw demonstrations to the LeRobot-style format.
2. Create or edit a dataset YAML under `dataset/`.
3. Train the geometric memory policy in simulation with ground-truth slots.
4. Fine-tune the simulation policy while still using ground-truth slots.
5. Train a simulation bridge that switches the policy input to 15D robot state
   and learns to predict slots from visual tokens using simulator slot labels.
6. Fine-tune on real-robot data with 15D robot state, predicted slots, and
   frozen slot-memory modules.

Example data conversion:

```bash
python scripts/convert_rmb_hsr_to_lerobot.py \
  --input /path/to/raw_demo_dir \
  --output /path/to/lerobot_dataset \
  --task-text "Pick up the can and place it into the box." \
  --include-hand-depth \
  --hand-depth-encoding inverse
```

Recommended training scripts:

```bash
bash scripts/train_01_sim_gt_slots_stage1_state27.sh
bash scripts/train_02_sim_gt_slots_stage2_state27.sh
bash scripts/train_03_sim_predslots_bridge_state15.sh
bash scripts/train_04_real_state15_finetune_action_vision.sh
```

The first two scripts use `state27 = robot_state15 + 4 * [dx, dy, valid]` so
that the geometric memory module can learn from accurate simulator slots. The
third script uses only robot-state15 as policy input and trains
`VisualSceneSlotPredictor` with simulator slots as auxiliary labels. The fourth
script is the real-robot adaptation stage: it keeps the slot predictor and
geometric memory fixed by default, while adapting the action pathway and the
vision branch to the real domain.

The shell scripts are examples and templates. For a new task, update the dataset
YAML, checkpoint paths, task text, state dimension, image-view mapping, and slot
options as needed.

For a quick smoke run on the public demo dataset, use fewer steps:

```bash
export PRETRAIN_CKPT=/path/to/Evo1_LIBERO_STD
export DATASET_CONFIG=dataset/config_demo_hsr_tomato_randomrect_realmatch_mix80_state27.yaml

MAX_STEPS=200 \
CKPT_INTERVAL=100 \
RUN_NAME=demo_stage1_state27_smoke \
bash scripts/train_01_sim_gt_slots_stage1_state27.sh
```

For a normal state27 simulation run:

```bash
export PRETRAIN_CKPT=/path/to/Evo1_LIBERO_STD
export DATASET_CONFIG=dataset/config_demo_hsr_tomato_randomrect_realmatch_mix80_state27.yaml

bash scripts/train_01_sim_gt_slots_stage1_state27.sh
bash scripts/train_02_sim_gt_slots_stage2_state27.sh
```

### Simulation Rollout

Rollout is a two-process setup:

1. Start the Evo-Mibo WebSocket inference server.
2. Run the HSR MuJoCo rollout client.

Start the server from a trained checkpoint:

```bash
python scripts/Evo1_server.py \
  --ckpt_dir checkpoints/hsr_sim_gt_slots_stage2_state27_bs8_h50_img448/step_best \
  --port 9017 \
  --num_inference_timesteps 32
```

In another terminal, run five fixed-seed tomato-can-box rollouts:

```bash
export PYTHONPATH=/path/to/RoboManipBaselines:${PYTHONPATH}

python scripts/rollout_hsr_tomato_can_box.py \
  --task tomato_can_box_random \
  --env_id robo_manip_baselines/MujocoHsrTomatoCanBoxEnv-v0 \
  --uri ws://127.0.0.1:9017 \
  --seed_list 11,22,33,44,55 \
  --world_idx_list 0,1,2,3,4 \
  --prompt "Pick up the can and move it to the box." \
  --state_layout no_wrench15_slots4x3 \
  --slot_source ground_truth \
  --third_view hand_depth \
  --image_mask_mode rgbd \
  --exec_horizon 50 \
  --save_video \
  --out_dir rollouts/demo_state27
```

For a checkpoint trained with predicted internal slots and 15D robot state, use:

```bash
python scripts/rollout_hsr_tomato_can_box.py \
  --task tomato_can_box_random \
  --env_id robo_manip_baselines/MujocoHsrTomatoCanBoxEnv-v0 \
  --uri ws://127.0.0.1:9017 \
  --seed_list 11,22,33,44,55 \
  --world_idx_list 0,1,2,3,4 \
  --prompt "Pick up the can and move it to the box." \
  --state_layout no_wrench15 \
  --third_view hand_depth \
  --image_mask_mode rgbd \
  --exec_horizon 50 \
  --save_video \
  --out_dir rollouts/demo_state15_predslots
```

Rollout outputs are written under `rollouts/` and include per-episode summaries,
trajectory arrays, plots, and optional videos. Success is read from the MuJoCo
environment reward: an episode is counted as successful when `max_reward >= 1.0`.

## Included and Excluded Files

Included:

- Core model code
- Dataset loading and conversion code
- Training entrypoints and example training scripts
- One demo MuJoCo rollout script for the HSR tomato-can-box task
- DeepSpeed configs
- Notes describing the slot-memory design

Excluded:

- Checkpoints and pretrained weights
- Raw datasets
- Logs and experiment outputs
- Rollout videos
- Generated evaluation results
- The previous project README

## Notes for New Tasks

This code is not limited to a single tomato-can task. To train another task,
prepare demonstrations with:

- language instruction
- synchronized image observations
- robot state
- action chunks or per-step actions
- optional inverse-depth view
- optional scene-slot targets

If no slot supervision is available, the model can be trained in robot-state-only
or image-plus-state mode. If simulator slot labels are available, they can be
used to train the slot predictor and geometric memory pathway.
