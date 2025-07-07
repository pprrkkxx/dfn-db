# 📡 Signal Dataset Processing and Training

## 📁 Folder Structure

```
project_root/
├── signalv3/                 # Processed detection dataset
│   ├── *.h5                  # HDF5 files with signals and labels
│
├── train_mat/               # Contains raw Hiarmod2019.1 dataset
│   └── train_mat.zip        # Unzip to get 'train_mat.h5'
│
├── gen_val_detection_v2.py  # Script to generate detection dataset
├── datasetdrawing.py        # Visualization of signals
├── train_log/               # Training results and weights
│   ├── *.pt                 # Model weight files
│   └── *.txt / *.log        # Training logs
```

---

## 📊 Dataset Details

The `signalv3` folder contains labeled signal samples and their corresponding spectrograms generated from the **Hiarmod2019.1** dataset.

### 🔉 SNR Mapping (Index to SNR)

Each signal sample is indexed and assigned a specific Signal-to-Noise Ratio (SNR) as follows:

| Index Range | SNR (dB) |
|-------------|----------|
| 0–499       | 6        |
| 500–999     | 4        |
| 1000–1499   | 2        |
| 1500–1999   | 0        |
| 2000–2499   | -2       |
| 2500–2999   | -4       |
| 3000–3499   | -6       |
| 3500–3999   | -8       |
| 4000–4499   | -10      |
| 4500–4999   | -12      |

---

## 🧰 How to Generate the Dataset

### Step 1: Unzip raw dataset

```bash
cd train_mat
unzip train_mat.zip
```

This creates: `train_mat/train_mat.h5`

---

### Step 2: Generate processed detection dataset

```bash
python gen_val_detection_v2.py
```

This will generate the processed dataset inside the `signalv3/` folder.

---

## ⚙️ Customization Options

You can edit `gen_val_detection_v2.py` to control dataset generation:

- **💾 Only save images (skip raw data):**  
  Comment out **line 336** and **line 385**

- **📶 Change SNR for generated signals:**  
  Edit **line 35** and **line 361**

- **🖼️ Control how many samples/images to generate:**  
  Adjust the `for` loop range at **line 341**

---

## 🖼️ Visualizing Signal Data

Run the following to visualize the time-domain and frequency-domain characteristics of generated signals:

```bash
python datasetdrawing.py
```

---

## 📈 Training Results

The `train_log/` folder contains:

- ✅ Trained model weights (`*.pt`)
- 📋 Training logs (`*.txt`, `*.log`)
- 📊 Experimental results under various SNR levels

---

## 📌 Summary

| Task                        | File / Folder               |
|-----------------------------|-----------------------------|
| Generate detection dataset  | `gen_val_detection_v2.py`   |
| Visualize signal processing | `datasetdrawing.py`         |
| Raw signal dataset          | `train_mat/train_mat.h5`    |
| Final dataset               | `signalv3/`                 |
| Logs & checkpoints          | `train_log/`                |

---

> For any further clarification or automation (e.g., batch generation or SNR loops), feel free to request assistance.