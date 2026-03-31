# Structural Anchors & Type Adaptation for Instruction-Driven Image Editing

Official code for instruction-driven image editing with structural grounding.

This repository provides a structure-aware image editing pipeline for multi-object scenes.  
Given an input image, a scene graph, and instance masks, it performs target-specific editing based on natural language instructions.

## Overview

<p align="center">
  <img src="fig/teaser.png" width="96%">
</p>

<p align="center">
  <img src="fig/framework_overview.png" width="96%">
</p>

Supported edit types:

- attribute edit
- object removal
- object replacement
- relation edit
- object addition

## Requirements

- Python >= 3.9
- Linux recommended
- CUDA-capable GPU recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation

This code assumes the following inputs are already available:

- input image
- scene graph JSON
- instance masks

Expected layout:

```text
output/
  no_edit/
    63.jpg
  scene_graphs_vg/
    63_scene_graph.json
  masks_vg/
    63/
      63_obj1533910_mask.npy
```

Required paths:

```text
output/no_edit/{image_id}.jpg
output/scene_graphs_vg/{image_id}_scene_graph.json
output/masks_vg/{image_id}/{image_id}_obj{obj_id}_mask.npy
```

## Run

Main entry point:

```bash
python -m scripts.graph_delta.cli
```

Common arguments:

- `--image_id`: image ID
- `--prompt_tgt`: editing instruction
- `--graph_root`: path to scene graph files
- `--mask_mode`: mask mode
- `--sd_inpaint_model`: path to inpainting model
- `--clip_local_path`: path to local CLIP model

### Examples

Remove:

```bash
python -m scripts.graph_delta.cli \
  --image_id 584 \
  --prompt_tgt "remove the left elephant" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

Attribute edit:

```bash
python -m scripts.graph_delta.cli \
  --image_id 555 \
  --prompt_tgt "change the pot on the table to red" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

Replace:

```bash
python -m scripts.graph_delta.cli \
  --image_id 63 \
  --prompt_tgt "replace the left knife with a fork" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

Relation edit:

```bash
python -m scripts.graph_delta.cli \
  --image_id 132 \
  --prompt_tgt "move the car to the left of the grass" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

Add:

```bash
python -m scripts.graph_delta.cli \
  --image_id 189 \
  --prompt_tgt "add a lion to the left of the lion" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

## Supported edit types

- attribute: `change the pot on the table to red`
- remove: `remove the left elephant`
- replace: `replace the left knife with a fork`
- relation: `move the car to the left of the grass`
- add: `add a lion to the left of the lion`

Note: `replace` is currently implemented as `remove + add`. See `scripts/graph_delta/pipeline_replace.py`.

## Project structure

```text
SGGE_DM/
├── diffusion/
├── graph/
├── scripts/graph_delta/
├── output/
└── requirements.txt
```

Main components:

- `scripts/graph_delta/cli.py`: command-line entry
- `scripts/graph_delta/parse_intent.py`: instruction parsing
- `graph/graph_delta.py`: graph-based edit planning
- `graph/graph_tokens.py`: scene graph representation
- `graph/mask_utils.py`: mask utilities
- `scripts/graph_delta/pipeline_attribute.py`: attribute editing
- `scripts/graph_delta/pipeline_remove.py`: remove editing
- `scripts/graph_delta/pipeline_replace.py`: replace editing
- `scripts/graph_delta/pipeline_relation.py`: relation editing
- `scripts/graph_delta/pipeline_add.py`: add editing

## Output

Edited results are typically saved under:

```text
output/graph_delta_instruct_edits/
```

## Limitations

- scene graphs and instance masks must be provided
- this is not a full end-to-end perception pipeline
- some instruction parsing is rule-based
- replace is currently implemented as remove + add

## Citation

```bibtex
@inproceedings{anonymous2026structuralanchors,
  title={Structural Anchors and Type Adaptation: Enabling Stable and Controllable Instruction-Driven Image Editing},
  author={Anonymous Authors},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```
