import sys
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModel

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

class VoiceDetector:
    def __init__(self):
        print("Loading multilingual speech model...")
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()

    def load_audio(self, path):
        audio, sr = librosa.load(path, sr=16000, mono=True)
        return audio, sr

    def predict(self, path):
        audio, sr = self.load_audio(path)

        inputs = self.extractor(audio, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

        # Simple heuristic score (demo deepfake detection)
        score = torch.mean(torch.abs(embeddings)).item()

        if score > 0.6:
            label = "AI_GENERATED"
        else:
            label = "HUMAN"

        confidence = min(score, 1.0)

        return label, confidence


def main():
    if len(sys.argv) < 2:
        print("Usage: python detector.py file.mp3")
        return

    file_path = sys.argv[1]

    detector = VoiceDetector()
    label, confidence = detector.predict(file_path)

    print(f"{label} ({confidence:.2f})")


if __name__ == "__main__":
    main()
