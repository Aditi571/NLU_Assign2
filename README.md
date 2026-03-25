# IIT Jodhpur NLP Project

## Setup Virtual Environment

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

---

## Running Tasks

### Problem 1

Navigate to the `prob1` folder:

```bash
cd prob1
```

Run the task:

```bash
python task1.py
```

---

### Problem 2

Navigate to the `prob2` folder:

```bash
cd ../prob2
```

Run the task:

```bash
python task2.py
```

---

### Problem 3 (RNN / BLSTM / Attention Models)

Navigate to the `prob3` folder:

```bash
cd ../prob3
```

Run the training script:

```bash
python task3.py
```

The script will:

- Train Vanilla RNN, Bidirectional LSTM (BLSTM), and Attention RNN models
- Print number of parameters and training loss per epoch
- Generate a training loss plot comparing all three models
- Save trained models as:
  - `vanilla_rnn.pth`
  - `blstm.pth`
  - `attention_rnn.pth`

---

## Notes

- Always make sure the virtual environment is activated before running any task
- Dependencies are managed in `requirements.txt`
- To view training progress for Problem 3, the matplotlib plot will show loss curves for all models
- You can load the trained models later using:

```python
model.load_state_dict(torch.load("model_name.pth"))
```

---

## Project Structure

```
project/
│
├── prob1/
│   └── task1.py
├── prob2/
│   └── task2.py
├── prob3/
│   └── task3.py
│   └── TrainingNames.txt
├── requirements.txt
└── README.md
```