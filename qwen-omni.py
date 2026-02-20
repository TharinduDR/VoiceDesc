from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import librosa
import torch

model_name = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

audio_path = "my_speech.wav"
audio, sr = librosa.load(audio_path, sr=16000)

prompt = (
    "Describe this speech in detail, including the language spoken, "
    "speaker characteristics, emotion, and content."
)

# Use the default system prompt (avoids the warning)
conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ],
    },
]

text = processor.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True
)

# Build inputs manually since processor ignores 'audios'
# 1. Tokenize text
text_inputs = processor.tokenizer(
    [text], return_tensors="pt", padding=True
)

# 2. Process audio via the feature extractor
audio_inputs = processor.feature_extractor(
    [audio], sampling_rate=16000, return_tensors="pt"
)

# 3. Combine
inputs = {**text_inputs, **audio_inputs}
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate() returns a tuple: (text_ids, audio_ids)
# We only need the text output
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    return_audio=False,  # text only, skip audio generation
)

# Handle both tuple and tensor returns
if isinstance(outputs, tuple):
    text_ids = outputs[0]
else:
    text_ids = outputs

generated_ids = text_ids[:, inputs["input_ids"].size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("=" * 60)
print("Description:")
print(response)