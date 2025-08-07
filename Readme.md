# Readme

This document contains part of the source code for the paper *"SmartGen: Synthesizing Context-Aware User Behavior Data for Adaptive Smart Home Intelligence"*. It includes four functional modules (TSS, SSC, GSS, TOF), data synthesis procedures, testing on two smart home tasks, parameter experiments, and ablation studies. Please note the following:

1. In the code, **SPPC** refers to **SSC**. SPPC is an earlier naming version that was not changed due to the established workflow.
2. The **SmartGen** folder contains the implementations of the four functional modules as well as the data synthesis system. The other six folders correspond to various experimental setups.
3. Each folder includes a `main.py` file, which serves as the entry point for running the corresponding experiment. The CSV files in the `results` directory contain the recorded outcomes of these experiments.
4. This code does **not** provide APIs for large language models, but it **does** open-source various types of synthesized data and synthesis logs.

Recommended compression thresholds in SmartGen:
| Dataset | Original Context | New Context | Compression Threshold | Anomaly Detection Percentage |
| ------- | ---------------- | ----------- | --------------------- | ---------------------------- |
|         | winter           | spring      | 0.918                 | 95.5                         |
| FR      | daytime          | night       | 0.92                  | 95                           |
|         | single           | multiple    | 0.915                 | 99                           |
|         | winter           | spring      | 0.915                 | 95                           |
| SP      | daytime          | night       | 0.917                 | 95                           |
|         | single           | multiple    | 0.915                 | 99                           |
|         | winter           | spring      | 0.905                 | 95                           |
| US      | daytime          | night       | 0.919                 | 93                           |
|         | single           | multiple    | 0.913                 | 99                           |


