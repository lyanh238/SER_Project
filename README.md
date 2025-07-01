# Speech Emotion Recognition using CNN-BiLSTM

This project implements a Speech Emotion Recognition (SER) system using a hybrid Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) architecture. The system is trained and evaluated on popular emotional speech datasets including:

* **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
* **EMO-DB**: Berlin Database of Emotional Speech
* **TESS**: Toronto Emotional Speech Set

## Project Structure

```
SER_Project/
├── data/                # Raw audio datasets
├── features/            # Preprocessed feature files
├── models/              # Saved trained models
├── outputs/             # Training logs, visualizations, confusion matrices
├── utils/               # Helper functions (data loader, feature extraction, etc.)
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── evaluation.py
│   └── visualization.py
├── model/               # Model architecture
│   └── cnn_bilstm.py
├── train.py             # Model training script
├── test.py              # Model evaluation script
├── inference.py         # Real-time prediction or batch inference
├── requirements.txt     # List of required Python libraries
└── README.md            # Project documentation
```

## Features

* Robust feature extraction using MFCCs, delta, delta-delta, RMS energy, and zero-crossing rate.
* CNN layers to capture local spatial features from audio signals.
* BiLSTM layers to capture sequential dependencies.
* Weighted sampling to handle class imbalance.
* Performance tracking using Accuracy and F1-score.
* Visualization of training progress and confusion matrices.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/lyanh238/SER_Project.git
cd SER_Project
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download datasets using KaggleHub or manually place them in the `data/` folder.

## Training

```bash
python train.py
```

## Evaluation

```bash
python test.py
```

## Inference

```bash
python inference.py
```

## Dependencies

* Python 3.8+
* PyTorch
* Librosa
* Scikit-learn
* Matplotlib
* Seaborn
* Soundfile
* TQDM
* KaggleHub

## License

This project is licensed under the MIT License.

## Contact

For questions or collaboration:

* **Name:** Your Name
* **Email:** [lenguyenquocanh2005@gmail.com](lenguyenquocanh2005@gmail.com)
