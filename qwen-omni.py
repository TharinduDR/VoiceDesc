from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import librosa
import inspect

model_name = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

# DEBUG: Print what the processor actually accepts
sig = inspect.signature(processor.__call__)
print("Processor __call__ params:", list(sig.parameters.keys()))
print("Processor type:", type(processor))
print("Has feature_extractor:", hasattr(processor, 'feature_extractor'))
print("Feature extractor type:", type(processor.feature_extractor))
print()

audio_path = "my_speech.wav"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio loaded: {len(audio)} samples, {len(audio)/sr:.1f}s, sr={sr}")

prompt = (
    "Describe the speaker's voice. Provide as many qualities of the voice "
    "as possible, and make them as detailed as possible."
)

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
print("Chat template text (first 500 chars):")
print(text[:500])
print()

# Process audio explicitly through the feature extractor
audio_features = processor.feature_extractor(
    [audio], sampling_rate=16000, return_tensors="pt"
)
print("Audio feature keys:", list(audio_features.keys()))
for k, v in audio_features.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# Tokenize text
text_inputs = processor.tokenizer(
    [text], return_tensors="pt", padding=True
)
print("Text input keys:", list(text_inputs.keys()))

# Combine everything
inputs = {**text_inputs, **audio_features}
inputs = {k: v.to(model.device) for k, v in inputs.items()}
print("Combined input keys:", list(inputs.keys()))

output = model.generate(**inputs, max_new_tokens=256, return_audio=False)

if isinstance(output, tuple):
    text_ids = output[0]
else:
    text_ids = output

generated_ids = text_ids[:, inputs["input_ids"].size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("=" * 60)
print("Description:")
print(response)