# MSJC Project: Hierarchical Jamming Classification in 5G O-RAN

## 1. Project Context
This project implements the MSJC framework on the CCI xG Testbed using 3 distributed instances and 3 USRPs. The goal is to detect and classify jamming attacks (Constant, Reactive, Protocol-aware) with a 94.51% accuracy target.

## 2. Infrastructure & Hardware Mapping
- **Gateway:** 38.68.231.178 (User: henrykim18345674)
- **Nodes & USRPs:**
  - **Instance-1 (UE):** 172.167.2.208 | USRP: 192.168.114.2 (Serial: 323DF47)
  - **Instance-2 (Classifier/RIC):** 172.167.1.165 | USRP: 192.168.116.2 (Serial: 323DF42)
  - **Instance-3 (Jammer):** 172.167.1.244 | USRP: 192.168.115.2 (Serial: 323EE55)

## 3. Technology Stack
- **Languages:** Python (PyTorch, Scikit-learn, FastAPI, Asyncio)
- **SDR Control:** UHD (USRP Hardware Driver) Python API
- **Data & MLOps:** InfluxDB (KPI logging), ClearML (Model management & retraining)

## 4. MSJC Pipeline Architecture
1. **Stage 1 (MLP):** Fast binary anomaly detection (Normal vs. Jammed).
2. **Stage 2 (KSVM):** Behavioral classification of physical jamming signatures.
3. **Stage 3 (MobileNetV3):** Semantic analysis of protocol-aware attacks using 2D spectrograms ($224 \times 224$).

## 5. Development Constraints
- **Latency:** Inference must complete within O-RAN Near-RT RIC timeframes (10ms - 1s).
- **Networking:** MTU 9000 is required for USRP I/Q streaming on `ens5`.
- **Atomic Updates:** Implement 'In-place model update' logic to switch model weights without service interruption.

## 6. Build & Test Commands
- **Check Hardware:** `uhd_find_devices`
- **Run UE (Instance-1):** `python3 ue_transmitter.py`
- **Run Jammer (Instance-3):** `python3 jammer.py`
- **Run MSJC Pipeline (Instance-2):** `python3 pipeline_runner.py`