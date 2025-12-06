# ID2223-Lab2

**Course:** ID2223 Scalable Machine Learning and Deep Learning
**Project:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA
**Tools:** PyTorch, Unsloth, Hugging Face, Gradio

## Project Overview

This repository contains our solution for Lab 2. The goal of this project was to construct a complete LLM pipeline: starting from fine-tuning a pre-trained foundation model on a specific instruction dataset, to deploying a scalable, serverless user interface (UI) for inference. And also try to think how to improve the model from the initial version.

We utilized the Unsloth library to optimize training on limited hardware (Google Colab T4 GPU). After running out of the free usage of Colab, we rented a GPU to train the model. And we deployed the final model on Hugging Face Spaces using CPU inference.

---

## Task 1: The Pipeline & Demo

In the first phase, we focused on establishing a working pipeline. We selected a lightweight model to ensure it could be trained within the free GPU time limits and run efficiently on a CPU-based inference server.

### Pipeline Steps:
### Step 1: Environment & Dependencies
We leverage `unsloth` for its optimized kernels which allow for 2x faster training and 60% less memory usage compared to standard HF Transformers.

```python
%%capture
!pip install unsloth
# Installs the nightly version of Unsloth for Llama-3.2 support
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+[https://github.com/unslothai/unsloth.git@nightly](https://github.com/unslothai/unsloth.git@nightly) git+[https://github.com/unslothai/unsloth-zoo.git](https://github.com/unslothai/unsloth-zoo.git)
```

### Step 2: Model Initialization (Architecture Selection)
For our improved pipeline (Task 2), we specifically selected the **Llama-3.2-3B-Instruct** model instead of the 1B baseline. We load it in 4-bit quantization to ensure it fits within the 16GB VRAM limit of Google Colab's T4 GPU.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024 
dtype = None 
load_in_4bit = True 

# Task 2 Improvement: Loading the 3B parameter model for better reasoning capabilities
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### Step 3: PEFT/LoRA Adapter Configuration
This is the core of our fine-tuning strategy. To improve model performance (Task 2), we configured the Low-Rank Adaptation (LoRA) with a higher rank (`r=16`) and alpha (`alpha=32`) to increase the model's capacity for learning new instructions.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Task 2 Improvement: Increased from standard 8 to 16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, # Task 2 Improvement: Scaled alpha (2x rank) for stable updates
    lora_dropout = 0, 
    bias = "none",   
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)
```

### Step 4: Data Processing
We used the `FineTome-100k` dataset, a high-quality instruction-following dataset. We also utilized `standardize_sharegpt` to ensure the data format aligns with Llama-3's chat template.

```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)
```

### Step 5: Training & Export
The model was trained using the `SFTTrainer` and finally exported to **GGUF format (q4_k_m)**. This specific quantization format is crucial for running the model efficiently on the CPU-only hardware available in Hugging Face Spaces.

```python
# Exporting to GGUF for CPU Inference
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

### The chatbot
**Try our Chatbot here:**  [Chatbot](https://huggingface.co/spaces/Yilai228/ID2223_Lab2)

---

## Task 2: Model Performance Improvement (Model-Centric Approach)

For the second task, we aimed to improve the model's performance. According to the course requirements, we adopted a model-centric approach.

We identified a trade-off between inference latency (speed) and reasoning quality (intelligence). Our goal was to determine if upgrading the model architecture and adapter complexity would yield better responses, even with the constraint of CPU inference.

### 1. The Experiment Setup

We designed a comparison experiment between a "Baseline" model (optimized for speed) and an "Improved" model (optimized for quality).

| Configuration | Baseline Model (Task 1) | Improved Model (Task 2) | Rationale for Change |
| :--- | :--- | :--- | :--- |
| **Base Model** | `Llama-3.2-1B-Instruct` | **`Llama-3.2-3B-Instruct`** | We upgraded to the 3B parameter model to capture more world knowledge and improve logic, as 1B models can struggle with complex instructions. |
| **LoRA Rank (r)** | 8 | **16** | We increased the rank from 8 to 16. A higher rank implies a larger number of trainable parameters, allowing the adapter to learn more subtle details from the dataset. |
| **LoRA Alpha** | 16 | **32** | We adjusted the scaling factor alpha to 32 to maintain a consistent ratio with the rank, ensuring stable training dynamics. |



### 2. Experimental Results & Analysis

After training both models, we conducted a side-by-side comparison on Hugging Face Spaces.

#### A. Inference Speed (Latency)
* **Observation:** The 1B Baseline model is significantly faster on CPU, generating tokens almost instantly. The 3B Improved model is slower due to the increased computational cost of the larger architecture.
* **Conclusion:** For real-time applications where speed is critical, the 1B model is superior.

#### B. Response Quality
* **Observation:** The 3B Improved model demonstrates superior instruction-following capabilities.
* **Conclusion:** The 3B model produces responses that are more coherent, follow instructions better, and hallucinate less.

### Qualitative Comparison (Sample Outputs)

Below are the actual outputs generated by both models for the same prompts, demonstrating the improvement in reasoning capabilities.

#### Prompt 1: Logic & Math
> **User:** Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,

**Baseline (1B Model):**
> The Fibonacci sequence is a series of numbers in which each number is the sum of the two preceding ones, usually starting with 0 and 1.\n2. The Fibonacci sequence has two popular starting numbers: 0 and 1. The numbers are called 0-nb, 1-nb,

*Comment: It fails to simply continue the sequence and starts explaining definitions unrelatedly*

**Improved (3B Model):**
> The Fibonacci sequence is a series of numbers where a number is the addition of the last two numbers, starting with 0 and 1. The sequence continues as follows:\n\n1, 1, 2, 3, 5, 8, 13, 21, 34, 55,

*Comment: Successfully continues the sequence*

---

#### Prompt 2: Descriptive Writing
> **User:** Describe a tall tower in the capital of France.

**Baseline (1B Model):**
> A tall tower in the capital of France can be a very imposing structure. This tower would likely stand at around 70-80 meters tall and could be a large building such as a cathedral or government center. The tower would be a striking feature of the city's landscape, visible from many different angles. Its height and grandeur would give a sense of the tower's importance to the city. The tower might also have a historical or cultural significance. For example, the tower could be a relic of a significant event in the country's history or the tower's design could have a special architectural significance. Overall, a tall tower in the

*Comment: Generic description, hallucinates the height*

**Improved (3B Model):**
> In the capital of France, there stands a magnificent tall tower. Located at the heart of the city, this stunning tower offers breathtaking views of the surrounding landscape. With its elegant, modern design, the tower stands out among other iconic structures and buildings. The tower stands to be about 160 metres tall and it is a testament to French architectural genius. This building is covered with intricate designs that reflect the country's rich history and heritage. The tower houses a museum dedicated to showcasing the history of the city and its inhabitants. The tower stands at the center of the city's major district and it is an important landmark that is visible from all areas

*Comment: More detailed, coherent, and "story-like" description*

### Conclusion
We concluded that for a Chatbot application where user engagement depends on response quality, the **gain in intelligence with the 3B model justifies the moderate increase in latency**.

### 3. Conclusion

Through this lab, we successfully demonstrated that a model-centric approach—specifically increasing model size and LoRA adapter rank—can significantly enhance the qualitative performance of an LLM.

While the 3B model (Rank 16) incurs a cost in terms of inference speed on CPU hardware, we believe the improvement in response accuracy and coherence justifies this trade-off for use cases requiring higher intelligence.

---

## How to Reproduce

To replicate our training process:

1.  Open the `.ipynb` notebook in Google Colab (select T4 GPU).
2.  Install dependencies: `pip install unsloth`.
3.  Select the desired configuration (1B or 3B model).
4.  Run the training loop and export to GGUF.

