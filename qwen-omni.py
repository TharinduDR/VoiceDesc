from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import os

model_name = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

# Use ABSOLUTE path — this is critical
audio_path = os.path.abspath("my_speech.wav")
print(f"Audio path: {audio_path}")
print(f"File exists: {os.path.exists(audio_path)}")

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

# Step 1: Render chat template as text
text = processor.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True
)

# Step 2: Use process_mm_info to properly load and process audio
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

print(f"Number of audio clips: {len(audios)}")
if audios:
    print(f"Audio array shape: {audios[0].shape}, dtype: {audios[0].dtype}")

# Step 3: Pass through processor with audio= (SINGULAR)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
)

# Debug: check token count — should be MUCH more than 915
print(f"Input keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}")

inputs = inputs.to(model.device)

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