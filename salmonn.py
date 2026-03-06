import os
import sys

# Add SALMONN repo to path
sys.path.insert(0, os.environ["SALMONN_REPO"])

from models.salmonn import SALMONN

print("Loading SALMONN...")
model = SALMONN(
    ckpt=os.environ["SALMONN_CKPT"],
    whisper_path=os.environ["WHISPER_PATH"],
    beats_path=os.environ["BEATS_PATH"],
    vicuna_path=os.environ["VICUNA_PATH"],
    low_resource=True,  # 8-bit quantisation, remove if you have 40GB+ VRAM
)
print("Model loaded!")

# Single file test
audio_path = "my_speech.wav"
prompt = (
    "Describe the speaker's voice. Provide as many qualities of the voice as possible, and make them as detailed as possible. The overall description should allow someone to be able to identify that voice by listening to it and identifying the same features."
)

response = model.generate(
    wav_path=audio_path,
    prompt=prompt,
    max_new_tokens=256,
    num_beams=4,
)

print("=" * 60)
print("Description:")
print(response)