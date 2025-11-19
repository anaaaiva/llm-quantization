# Quantized LLM Inference with Triton Kernels

## Project Overview

This project focuses on implementing **custom Triton** kernels for **weight quantization** and **inference of quantized large language models (LLMs)**. The goal is to reduce memory usage and improve computational efficiency while maintaining acceptable model performance.

## Objectives

1. **Quantization Kernel Implementation**
   Implement a kernel that quantizes a 2D matrix from **FP16** to **INT4**, and packs the quantized matrix into **INT8** or **INT32** format.
   The memory consumption should decrease by **4×** compared to the original FP16 representation.
   | quantization_type | group_size |    mae   | compression |
   |-------------------|------------|----------|-------------|
   | GLOBAL            | –          | 0.169577 | 3.999999    |
   | ROWWISE           | –          | 0.096904 | 3.996098    |
   | GROUPWISE         | 16         | 0.069842 | 3.200000    |
   | GROUPWISE         | 32         | 0.076343 | 3.555556    |
   | GROUPWISE         | 64         | 0.082926 | 3.764706    |
   | GROUPWISE         | 128        | 0.089447 | 3.878788    |
   | GROUPWISE         | 256        | 0.095765 | 3.938462    |
   | GROUPWISE         | 512        | 0.101913 | 3.968992    |
   | GROUPWISE         | 1024       | 0.107781 | 3.984436    |
   | GROUPWISE         | 2048       | 0.113451 | 3.992203    |
   | GROUPWISE         | 4096       | 0.118949 | 3.996098    |

2. **Matrix Multiplication Kernel**
   Implement a kernel for matrix multiplication between:

   * an activation matrix **X** in **BF16**, and
   * a quantized weight matrix **W** in **INT4**.

   Operation: `Y = X_bf16 @ W_int4^T`

3. **Performance Comparison**
   Compare the speed of the quantized matrix multiplication `(X_bf16 @ W_int4^T)` with the full-precision version `(X_bf16 @ W_bf16^T)`.
   Use matrix dimensions equivalent to the **weight matrices of the Llama-3.2-1B-Instruct** model ([link](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)).
   Test with different activation matrix sizes (number of tokens): **128**, **512**, and **2048**.

   | M    | mode     | KxN         |  time_ms    | speedup_vs_fp16 | mae_vs_fp16 | timedown_vs_fp16 |
   |------|----------|-------------|-------------|-----------------|------------|------------------|
   | 128  | global   | 128256x2048 |  90.718457  | 0.050633        | 63.031250  | 19.749852        |
   |      |          | 2048x2048   |   1.487995  | 0.060322        |  6.937500  | 16.577719        |
   |      |          | 2048x8192   |   5.039065  | 0.038293        |  7.003906  | 26.114212        |
   |      |          | 512x2048    |   0.419758  | 0.083549        |  3.259766  | 11.969013        |
   |      |          | 8192x2048   |   5.587930  | 0.034792        | 13.937500  | 28.742038        |
   |      | groupwise| 128256x2048 | 106.093975  | 0.043295        | 30.171875  | 23.097177        |
   |      |          | 2048x2048   |   1.489646  | 0.060255        |  3.810547  | 16.596116        |
   |      |          | 2048x8192   |   5.530464  | 0.034891        |  3.802734  | 28.660814        |
   |      |          | 512x2048    |   0.794894  | 0.044120        |  1.896484  | 22.665659        |
   |      |          | 8192x2048   |   6.939909  | 0.028014        |  7.605469  | 35.696066        |
   |      | rowwise  | 128256x2048 | 124.494805  | 0.036896        | 33.281250  | 27.103128        |
   |      |          | 2048x2048   |   1.816195  | 0.049421        |  4.179688  | 20.234186        |
   |      |          | 2048x8192   |   6.890054  | 0.028006        |  4.175781  | 35.706689        |
   |      |          | 512x2048    |   0.444377  | 0.078920        |  2.042969  | 12.670989        |
   |      |          | 8192x2048   |   7.415319  | 0.026218        |  8.304688  | 38.141382        |
   | 512  | global   | 128256x2048 | 353.944141  | 0.037931        | 61.500000  | 26.363610        |
   |      |          | 2048x2048   |   5.476802  | 0.025013        |  6.406250  | 39.979368        |
   |      |          | 2048x8192   |  21.177200  | 0.023120        |  7.117188  | 43.252727        |
   |      |          | 512x2048    |   1.478310  | 0.028608        |  3.167969  | 34.955496        |
   |      |          | 8192x2048   |  21.680942  | 0.046358        | 14.421875  | 21.571428        |
   |      | groupwise| 128256x2048 | 399.748984  | 0.033585        | 30.156250  | 29.775394        |
   |      |          | 2048x2048   |   5.913578  | 0.023165        |  3.804688  | 43.167731        |
   |      |          | 2048x8192   |  23.137310  | 0.021161        |  3.810547  | 47.256094        |
   |      |          | 512x2048    |   1.462065  | 0.028926        |  1.901367  | 34.571378        |
   |      |          | 8192x2048   |  26.927029  | 0.037326        |  7.605469  | 26.791015        |
   |      | rowwise  | 128256x2048 | 477.153789  | 0.028137        | 33.218750  | 35.540908        |
   |      |          | 2048x2048   |   7.289376  | 0.018793        |  4.183594  | 53.210730        |
   |      |          | 2048x8192   |  28.494343  | 0.017183        |  4.171875  | 58.197404        |
   |      |          | 512x2048    |   1.843914  | 0.022936        |  2.054688  | 43.600410        |
   |      |          | 8192x2048   |  29.629807  | 0.033921        |  8.304688  | 29.480142        |
   | 2048 | global   | 128256x2048 |1378.586875  | 0.037957        | 61.468750  | 26.345720        |
   |      |          | 2048x2048   |  21.333113  | 0.039414        |  6.699219  | 25.371751        |
   |      |          | 2048x8192   |  85.770586  | 0.036686        |  7.156250  | 27.258604        |
   |      |          | 512x2048    |   5.441779  | 0.022803        |  3.208984  | 43.854082        |
   |      |          | 8192x2048   |  85.745967  | 0.042918        | 14.187500  | 23.300334        |
   |      | groupwise| 128256x2048 |1533.772187  | 0.034116        | 30.156250  | 29.311415        |
   |      |          | 2048x2048   |  23.097856  | 0.036403        |  3.804688  | 27.470584        |
   |      |          | 2048x8192   |  92.468105  | 0.034028        |  3.810547  | 29.387132        |
   |      |          | 512x2048    |   5.659891  | 0.021924        |  1.901367  | 45.611794        |
   |      |          | 8192x2048   | 104.438848  | 0.035236        |  7.621094  | 28.379877        |
   |      | rowwise  | 128256x2048 |1840.048281  | 0.028438        | 33.281250  | 35.164557        |
   |      |          | 2048x2048   |  28.473135  | 0.029530        |  4.171875  | 33.863473        |
   |      |          | 2048x8192   | 113.715840  | 0.027670        |  4.171875  | 36.139838        |
   |      |          | 512x2048    |   7.126508  | 0.017412        |  2.052734  | 57.430931        |
   |      |          | 8192x2048   | 114.555908  | 0.032124        |  8.273438 | 31.129054
   
4. **Quantized Linear Layer Integration**
   Using the implemented kernels, create a **quantized linear layer** and integrate it into the **linear layers** of the Llama-3.2-1B-Instruct model.

5. **Evaluation**
   Measure:

   * **Inference speed** of the quantized model
      | mode     | tokens_per_sec | downgrade |
      |----------|----------------|-----------|
      | fp16     | 7761.302261    | 1.000000  |
      | global   | 331.114071     | 23.439965 |
      | rowwise  | 247.640887     | 31.340956 |
      | groupwise| 295.557271     | 26.259893 |
   * **Perplexity** on the **WikiText-2** dataset
      | mode     | perplexity   |
      |----------|--------------|
      | fp16     | 15.00816     |
      | global   | 1840133.0000 |
      | rowwise  | 31.38453     |
      | groupwise| 19.99439     |

All experiments were conducted on Google Colab.

