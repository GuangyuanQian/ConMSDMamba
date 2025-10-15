# ConMSDMamba: Multi-Scale Dilated Mamba based on Conformer for Speech Emotion Recognition

## How to Use

### 1. Feature Extraction

Before training, you need to extract acoustic features from the raw audio files. Use the `extract_features.py` script for this purpose.

You should run this script for each dataset you intend to use.

### 2. Model Training

Once the features have been extracted and saved, you can proceed to train the models. The training scripts are located within their respective dataset folders.

*   **To train the model on IEMOCAP:**
    ```bash
    cd IEMOCAP/
    python train.py
    ```

*   **To train the model on MELD:**
    ```bash
    cd MELD/
    python train.py
    ```
