# ID2223-Lab2

**Course:** ID2223 Scalable Machine Learning and Deep Learning
**Project:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA
**Tools:** PyTorch, Unsloth, Hugging Face, Gradio

## Project Overview

This repository contains our solution for Lab 2. The goal of this project was to construct a complete LLM pipeline: starting from fine-tuning a pre-trained foundation model on a specific instruction dataset, to deploying a scalable, serverless user interface (UI) for inference.

We utilized the Unsloth library to optimize training on limited hardware (Google Colab T4 GPU) and deployed the final model on Hugging Face Spaces using CPU inference.

---

## Task 1: The Pipeline & Demo

In the first phase, we focused on establishing a working pipeline. We selected a lightweight model to ensure it could be trained within the free GPU time limits and run efficiently on a CPU-based inference server.

### Pipeline Steps:
1.  **Data Preparation:** We used the FineTome-100k dataset, which provides high-quality instruction-response pairs.
2.  **Fine-Tuning:** We used QLoRA (4-bit Quantized LoRA) to fine-tune the `Llama-3.2-1B-Instruct` model. This allowed us to update the model weights efficiently without exceeding VRAM limits.
3.  **Export:** The trained model was merged and converted to GGUF format (`q4_k_m` quantization) to enable CPU inference.
4.  **Deployment:** We built a Gradio chat interface hosted on Hugging Face Spaces.

### Live Demo
**Try our Chatbot here:** 

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
* **Example Case:** When asked to "Explain the concept of recursion to a 5-year-old," the 3B model provided a more coherent analogy compared to the 1B model.

*(Place two screenshots here later: one showing the 1B response and one showing the 3B response)*

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

## Acknowledgements

* **Course:** ID2223 at KTH Royal Institute of Technology.
* **Library:** Unsloth AI for making fine-tuning accessible on free GPUs.
* **Dataset:** Maxime Labonne's FineTome dataset.
