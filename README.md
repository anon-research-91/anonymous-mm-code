# Structural Anchors & Type Adaptation for Instruction-Driven Image Editing

[![Paper](https://img.shields.io/badge/Paper-ACM_MM_2026-blue)](#)
[![Task](https://img.shields.io/badge/Task-Instruction--Driven_Image_Editing-green)](#)
[![Setting](https://img.shields.io/badge/Setting-Controlled_Study-orange)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Visual_Genome_/_OpenImages-lightgrey)](#)

Official code for the paper:

**Structural Anchors & Type Adaptation: Enabling Stable and Controllable Instruction-Driven Image Editing**

This repository presents a **structure-aware image editing pipeline** for complex multi-object scenes.  
Instead of directly applying global text-conditioned editing, the method first **grounds the instruction to scene entities and relations**, then performs **type-adaptive local editing**.

---

## Teaser

<p align="center">
  <img src="assets/teaser.png" width="92%">
</p>

Our method is designed for **target-specific instruction-driven image editing** with improved locality, structural consistency, and target grounding stability.

---

## Highlights

- **Structural anchors** for object- and relation-level grounding
- **Grounding-before-editing** for stable target identification
- **Type-adaptive execution** for attribute, remove, replace, relation, and add operations
- **Localized mask-based editing** for better background preservation
- A **controlled-study setting** with oracle scene graphs and instance masks
- Evaluation on **Visual Genome**, with additional qualitative transfer to **OpenImages V6**

---

## Framework Overview

<p align="center">
  <img src="assets/framework_overview.png" width="96%">
</p>

The method consists of three stages:

1. **Structure-aware semantic representation**
   - construct object- and relation-level anchors from the scene graph
   - parse the editing instruction into target, constraint, and edit type

2. **Edit localization and type inference**
   - identify the target object or object pair
   - generate a localized edit mask

3. **Type-adaptive edit execution**
   - choose an edit strategy according to the instruction type
   - apply editing while preserving non-target regions

---

## Why structure-aware editing?

Existing instruction-driven image editing methods often treat the instruction as a global conditioning signal.  
This can work for simple appearance changes, but becomes less reliable when:

- multiple similar object instances coexist in the scene
- the instruction contains spatial constraints such as **left**, **right**, or **next to**
- successful editing depends on **relation-aware target grounding**

Our method explicitly separates:

- **what to edit**
- **which instance to edit**
- **how to execute the edit**

This design improves controllability in complex multi-object scenes.

---

## Controlled Study Setting

<p align="center">
  <img src="assets/data_overview.png" width="92%">
</p>

This repository corresponds to the **controlled-study setting** used in the paper.

- It assumes that **scene graphs** and **instance masks** are already available
- It is **not** a full end-to-end perception-and-editing system
- The goal is to isolate the contribution of **explicit structural grounding**
- The main experiments are conducted under **oracle structural annotations**

---

## Method–Code Mapping

The paper uses the terminology **structural anchors** and **type adaptation**.  
In this repository, the corresponding code modules are:

| Paper Concept | Code |
|---|---|
| Scene-graph representation | `graph/graph_tokens.py`, `graph/mask_utils.py` |
| Instruction parsing / intent extraction | `scripts/graph_delta/parse_intent.py` |
| Graph-conditioned edit planning | `graph/graph_delta.py` |
| Semantic / graph-guided utilities | `graph/semantics.py`, `diffusion/graph_guided_attention.py` |
| Type-adaptive execution | `scripts/graph_delta/pipeline_attribute.py`, `pipeline_remove.py`, `pipeline_replace.py`, `pipeline_relation.py`, `pipeline_add.py` |

---

## Supported Edit Types

### Attribute Edit

Example:

```
Change the pot on the table to red.
```

### Remove

Example:

```
Remove the left elephant.
```

### Replace

Example:

```
Replace the left knife with a fork.
```

### Relation Edit

Example:

```
Move the car to the left of the grass.
```

### Add

Example:

```
Add a lion to the left of the lion.
```

**Implementation note:**
In the current codebase, **replace** is implemented as a two-stage procedure:

1. remove the source object  
2. add the target object at the source location  

See `scripts/graph_delta/pipeline_replace.py`.

---

## Repository Structure

```
SGGE_DM/
├── diffusion/
│   ├── blended_diffusion.py
│   ├── diffedit.py
│   ├── graph_guided_attention.py
│   └── instruct_pix2pix_editor.py
├── graph/
│   ├── graph_delta.py
│   ├── graph_tokens.py
│   ├── mask_utils.py
│   └── semantics.py
├── scripts/graph_delta/
│   ├── cli.py
│   ├── parse_intent.py
│   ├── pipeline_add.py
│   ├── pipeline_attribute.py
│   ├── pipeline_relation.py
│   ├── pipeline_remove.py
│   ├── pipeline_replace.py
│   └── ...
├── output/
│   ├── no_edit/
│   ├── scene_graphs_vg/
│   ├── masks_vg/
│   └── graph_delta_instruct_edits/
└── requirements.txt
```

---

## Setup

### Requirements

- Python >= 3.9  
- Linux recommended  
- CUDA-capable GPU recommended  

### Installation

```bash
pip install -r requirements.txt
```

---

## Data Layout

This repository assumes that **images**, **scene graphs**, and **instance masks** are already available locally.

Example layout:

```
output/
  no_edit/
    63.jpg
  scene_graphs_vg/
    63_scene_graph.json
  masks_vg/
    63/
      63_obj1533910_mask.npy
```

Required inputs:

```
output/no_edit/{image_id}.jpg
output/scene_graphs_vg/{image_id}_scene_graph.json
output/masks_vg/{image_id}/{image_id}_obj{obj_id}_mask.npy
```

If you use a different directory layout, pass the corresponding paths through the CLI arguments.

---

## Running the Code

Main entry point:

```bash
python -m scripts.graph_delta.cli
```

### Example: remove

```bash
python -m scripts.graph_delta.cli \
  --image_id 584 \
  --prompt_tgt "remove the left elephant" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

### Example: attribute edit

```bash
python -m scripts.graph_delta.cli \
  --image_id 555 \
  --prompt_tgt "change the pot on the table to red" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

### Example: replace

```bash
python -m scripts.graph_delta.cli \
  --image_id 63 \
  --prompt_tgt "replace the left knife with a fork" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

### Example: relation edit

```bash
python -m scripts.graph_delta.cli \
  --image_id 132 \
  --prompt_tgt "move the car to the left of the grass" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

### Example: add

```bash
python -m scripts.graph_delta.cli \
  --image_id 189 \
  --prompt_tgt "add a lion to the left of the lion" \
  --graph_root ./output/scene_graphs_vg \
  --mask_mode mask \
  --sd_inpaint_model /path/to/model \
  --clip_local_path /path/to/clip
```

---

## Qualitative Comparisons

<p align="center">
  <img src="assets/qualitative_comparison.png" width="96%">
</p>

Our method shows stronger target grounding stability and better preservation of non-target scene structure in complex multi-object editing scenarios.

---

## Ablation Study

<p align="center">
  <img src="assets/ablation.png" width="96%">
</p>

The ablation results highlight the importance of:

- explicit region restriction  
- structure-aware grounding  
- type-adaptive execution  

Removing any of these components degrades edit locality, target binding, or instruction faithfulness.

---

## Scope and Limitations

This repository corresponds to the **controlled-study setting** in the paper.

- It assumes **oracle scene graphs** and **instance masks**
- It does **not** include a full object detector, segmenter, or relation predictor
- Structural grounding quality depends on the provided scene graph and mask files
- Some operations use simplified rule-based instruction parsing
- `replace` is currently implemented as **remove + add**, not as a single native edit primitive

The purpose of this repository is to isolate and evaluate the contribution of **explicit structural grounding** in instruction-driven image editing.

---

## TODO

- [ ] Release cleaned inference scripts  
- [ ] Release example input/output assets  
- [ ] Add checkpoint preparation instructions  
- [ ] Add more evaluation scripts  
- [ ] Extend to end-to-end perception pipelines  

---

## Citation

```bibtex
@inproceedings{anonymous2026structuralanchors,
  title={Structural Anchors and Type Adaptation: Enabling Stable and Controllable Instruction-Driven Image Editing},
  author={Anonymous Authors},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```

---

## Acknowledgement

This project builds on open-source diffusion-based editing tools and public vision-language resources.
