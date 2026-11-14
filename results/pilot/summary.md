GPU: NVIDIA A100-PCIE-40GB MIG 2g.10gb, 10.47 GB, 28 SMs (0.259 of a full A100), measured bf16 matmul 65.4 TFLOP/s

| Run | Params (M) | Batch | Images/s (measured) | Peak GPU memory GB (measured) | Epochs | Hours per run (projected) | Runs | Hours total (projected) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | 11.17 | 128 | 3878.9 | 0.47 | 300 | 1.07 | 3 | 3.2 |
| vitb16_finetune | 85.81 | 32 | 230.7 | 3.97 | 5 | 0.3 | 1 | 0.3 |
| vitlite_convstem | 3.86 | 128 | 4182.7 | 0.65 | 300 | 1.0 | 3 | 3.0 |
| vitlite_p2 | 3.76 | 128 | 1576.6 | 2.06 | 300 | 2.64 | 3 | 7.9 |
| vitlite_p4 | 3.72 | 128 | 4615.6 | 0.57 | 300 | 0.9 | 3 | 2.7 |
| vitlite_p8 | 3.75 | 128 | 4677.0 | 0.21 | 300 | 0.89 | 3 | 2.7 |

Core total: 16.8 GPU-hours, about 101 SU (4.0% of the 2500 SU allocation)
Extension total: 3.0 GPU-hours, about 18 SU (0.7% of the 2500 SU allocation)
