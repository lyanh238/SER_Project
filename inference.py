import torch
from model.cnn_bilstm import CNNBiLSTM
from utils.feature_extraction import extract_features

label_to_int = {'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3, 'fearful': 4, 'disgust': 5, 'surprised': 6}
int_to_label = {v: k for k, v in label_to_int.items()}

model = CNNBiLSTM(len(label_to_int), 122, 173)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
model.eval()

def predict(audio_path):
    features = extract_features(audio_path)
    if features is None:
        return "Error processing audio."

    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs.data, 1)

    return int_to_label[predicted.item()]

# Example
print(predict('example_audio.wav'))