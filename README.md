# Quantized LLM Inference with Triton Kernels

## Project Overview

This project focuses on implementing **custom Triton** kernels for **weight quantization** and **inference of quantized large language models (LLMs)**. The goal is to reduce memory usage and improve computational efficiency while maintaining acceptable model performance.

## Objectives

1. **Quantization Kernel Implementation**
   Implement a kernel that quantizes a 2D matrix from **FP16** to **INT4**, and packs the quantized matrix into **INT8** or **INT32** format.
   The memory consumption should decrease by **4×** compared to the original FP16 representation.

2. **Matrix Multiplication Kernel**
   Implement a kernel for matrix multiplication between:

   * an activation matrix **X** in **BF16**, and
   * a quantized weight matrix **W** in **INT4**.

   Operation: `Y = X_bf16 @ W_int4^T`

3. **Performance Comparison**
   Compare the speed of the quantized matrix multiplication `(X_bf16 @ W_int4^T)` with the full-precision version `(X_bf16 @ W_bf16^T)`.
   Use matrix dimensions equivalent to the **weight matrices of the Llama-3.2-1B-Instruct** model ([link](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)).
   Test with different activation matrix sizes (number of tokens): **128**, **512**, and **2048**.

4. **Quantized Linear Layer Integration**
   Using the implemented kernels, create a **quantized linear layer** and integrate it into the **linear layers** of the Llama-3.2-1B-Instruct model.

5. **Evaluation**
   Measure:

   * **Inference speed** of the quantized model
   * **Perplexity** on the **WikiText-2** dataset

