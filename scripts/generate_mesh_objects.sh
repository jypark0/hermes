#!/bin/bash

# TYPES=("1" "12" "13" "123")
NUM_NODES=250
NUM_OBJECTS=50

SAVE_PATH="data/objects/N${NUM_NODES}_O${NUM_OBJECTS}/raw"

# Train
echo "TRAIN"
python src/data/objects/generate_mesh_objects.py \
  --env_id "MeshObjects-v0" \
  --env_kwargs \
  num_nodes="$NUM_NODES" num_objects="$NUM_OBJECTS" rules="123" \
  --num_episodes=20 \
  --env_timelimit=100 \
  --save_path="${SAVE_PATH}/train" \
  --seed=42

# Test
echo "TEST"
python src/data/objects/generate_mesh_objects.py \
  --env_id "MeshObjects-v0" \
  --env_kwargs \
  num_nodes="$NUM_NODES" num_objects="$NUM_OBJECTS" rules="123" \
  --reset_options random_hill_parameters=True \
  --num_episodes=100 \
  --env_timelimit=20 \
  --save_path="${SAVE_PATH}/test" \
  --seed=43
