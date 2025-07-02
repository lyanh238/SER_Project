import os
from tqdm import tqdm
import pandas as pd

all_audio_files = []

# RAVDESS dataset processing
ravdess_emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

ravdess_path = r"D:\SER_Project\data\ravdess-emotional-speech-audio"
ravdess_base_path = ravdess_path


for actor_dir in tqdm(os.listdir(ravdess_base_path), desc="Processing RAVDESS"):
    actor_path = os.path.join(ravdess_base_path, actor_dir)
    if os.path.isdir(actor_path):
        for filename in os.listdir(actor_path):
            if filename.endswith('.wav'):
                parts = filename.split('-')
                emotion_code = parts[2]
                emotion_name = ravdess_emotion_map.get(emotion_code)
                if emotion_name:
                    all_audio_files.append({
                        'filepath': os.path.join(actor_path, filename),
                        'emotion': emotion_name,
                        'dataset': 'RAVDESS'
                    })

# EMODB dataset processing
emodb_emotion_map = {
    'W': 'neutral', 'L': 'calm', 'H': 'happy', 'S': 'sad',
    'A': 'angry', 'F': 'fearful', 'T': 'disgust', 'N': 'surprised'
}

emodb_path = r"D:\SER_Project\data\berlin-database-of-emotional-speech-emodb"
emodb_base_path = emodb_path

for filename in tqdm(os.listdir(emodb_base_path), desc="Processing EmMODB"):
    if filename.endswith('.wav'):
        parts = filename.split('_')
        emotion_code = parts[2][0]  # Assuming the emotion code is the first character of the third part
        emotion_name = emodb_emotion_map.get(emotion_code)
        if emotion_name:
            all_audio_files.append({
                'filepath': os.path.join(emodb_base_path, filename),
                'emotion': emotion_name,
                'dataset': 'EmMODB'
            })

# TESS dataset processing
tess_emotion_map = {
    'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprise': 'surprised'
}

tess_path = r"D:\SER_Project\data\toronto-emotional-speech-set-tess"
tess_base_path = tess_path

for filename in tqdm(os.listdir(tess_base_path), desc="Processing TESS"):
    if filename.endswith('.wav'):
        parts = filename.split('_')
        emotion_name = parts[2].lower()  # Assuming the emotion is the third part
        if emotion_name in tess_emotion_map:
            all_audio_files.append({
                'filepath': os.path.join(tess_base_path, filename),
                'emotion': tess_emotion_map[emotion_name],
                'dataset': 'TESS'
            })

# Create a DataFrame and save to CSV
df = pd.DataFrame(all_audio_files) 

# Selected emotion, keep some emotion like "Happy", "Sad", "Angry", "Natural", "Fearful", "Disgust","Suprise"
selected_emotions = ['happy','sad','angry','natural','fearful','disgust','suprise']
df = df[df['emotion'].isin(selected_emotions)].reset_index(drop=True)


label_to_int = {emotion: i for i, emotion in enumerate(df['emotion'].unique())}
int_to_label = {i:emotion for emotion,i in label_to_int.items()}
df['label'] = df['emotion'].map(label_to_int)



df.to_csv("combined_datasets")