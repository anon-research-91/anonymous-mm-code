Anonymous Graph-Driven Image Editing (ACM MM Submission)

🔥 Highlights

·🧠 **Graph-driven editing**: Perform image editing via structured scene graph modifications
·🎯 **Precise region control**: Mask-guided editing enables localized and controllable modifications
·🧩 **Multi-task support**: Unified framework for addition, removal, replacement, attribute, and relation editing
·⚡ **Diffusion-based synthesis**: High-quality edits using modern diffusion models

🖼️ Overview

This repository provides the core implementation of a **graph-driven image editing pipeline**.

Given structured inputs such as graph deltas or editing instructions, our method performs **mask-aware, diffusion-based image editing** to produce semantically consistent results.

⚙️ Method Pipeline

Our pipeline consists of four key stages:

1. **Instruction Parsing**
   Convert graph deltas or text instructions into executable editing actions

2. **Task-specific Editing Modules**
   Handle different editing types:

   * Add / Remove / Replace
   * Attribute Editing
   * Relation Editing

3. **Mask Generation and Processing**
   Generate and refine spatial regions for editing

4. **Diffusion-based Editing**
   Perform localized image synthesis with mask guidance

🚀 Installation

```bash
git clone https://github.com/anonymous/anonymous-mm-code.git
cd anonymous-mm-code

pip install -r requirements.txt
```

---

## ⚡ Quick Demo

We provide a minimal working example:

```bash
python scripts/graph_delta/cli.py \
  --image example/image.png \
  --mask example/mask.png \
  --graph example/graph.json \
  --edit "replace object"
```

---

## 📂 Output Files

All results are saved in the `output/` directory, including:

* Edited images
* Intermediate mask visualizations
* JSON files describing graph structures or edits

These outputs illustrate both intermediate reasoning and final editing results.

---

## 📁 Project Structure

```text
anonymous-mm-code/
├── scripts/
│   └── graph_delta/
├── graph/
├── diffusion/
├── example/
├── output/
└── requirements.txt
```

---

## 📊 Data

Due to size and licensing restrictions, full datasets are not included.

Users can obtain datasets such as **Visual Genome** or **COCO** from official sources.

We provide small examples for demonstration purposes.

---

## 🔁 Reproducibility

* The example demonstrates the full pipeline
* Additional resources will be released after the review process

---

## ⚠️ Disclaimer

All identifying information has been removed to ensure compliance with double-blind review.

---

## 📜 License

Released for academic review purposes only.
