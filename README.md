# Image Caption Generator (CNN + GRU)

## Overview

This project builds an **image captioning model** that automatically generates natural language descriptions for images.

It combines:

* A **CNN (ResNet18)** to extract image features
* A **GRU-based RNN** to generate captions word-by-word

The model is trained on the **Flickr8k dataset**, which contains images paired with multiple captions.

---

## How It Works

### 1. Image Feature Extraction (CNN)

* Uses a pretrained **ResNet18**
* Removes the final classification layer
* Outputs a **256-dimensional feature vector** for each image

### 2. Caption Generation (GRU)

* Converts tokenized captions into embeddings
* Uses a **GRU (Gated Recurrent Unit)** to model sequence generation
* Predicts the next word at each time step

### 3. Training Flow

1. Image → CNN → feature vector
2. Feature vector initializes GRU hidden state
3. Caption tokens are shifted:

   * Input: `<SOS> ...`
   * Target: `... <EOS>`
4. GRU predicts next tokens
5. Loss computed using **CrossEntropyLoss**

---

## Dataset

* **Flickr8k**

  * ~8,000 images
  * Each image has multiple captions

Structure:

```
Flickr8k/
 ├── Images/
 └── captions.txt
```

---

## Installation

```bash
pip install torch torchvision transformers nltk matplotlib pillow
```

---

## Running the Project

1. Make sure dataset is in the correct directory:

```
Flickr8k/Images
Flickr8k/captions.txt
```

2. Run the script:

```bash
python main.py
```

---

## Model Architecture

### CNN Extractor

* Pretrained ResNet18
* Output: 256-dim vector

### GRU Decoder

* Embedding layer
* GRU (1 layer)
* Fully connected output → vocab size

---

## Key Hyperparameters

| Parameter       | Value |
| --------------- | ----- |
| Hidden Size     | 256   |
| Batch Size      | 64    |
| Epochs          | 10    |
| Learning Rate   | 0.001 |
| Max Caption Len | 30    |

---

## Evaluation

* Uses **BLEU Score** to measure caption quality
* A prediction is considered correct if:

```
BLEU ≥ 0.20
```

Final metric:

```
Accuracy = (Correct Predictions / Total Samples) × 100
```

---

## Example Output

**Input Image:**
(A random Flickr image)

**Generated Caption:**

```
"a dog running through the grass"
```

**Ground Truth:**

```
"a brown dog is running across a field"
```

---

##  Features

* End-to-end image captioning pipeline
* Uses pretrained CNN for better feature extraction
* Tokenization via HuggingFace tokenizer (T5)
* BLEU-based evaluation
* Custom Dataset + DataLoader

---
