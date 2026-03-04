from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import librosa
import os
import numpy as np

model_name = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

audio_path = os.path.abspath("my_speech.wav")
audio_np, sr = librosa.load(audio_path, sr=16000)

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

# Approach 1: apply_chat_template with tokenize=True and audio kwarg
print("Trying apply_chat_template with audio kwarg...")
try:
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        audio=[audio_np],
    )
    print("SUCCESS with audio kwarg")
except Exception as e:
    print(f"Failed: {e}")

    # Approach 2: embed numpy array directly in conversation
    print("\nTrying with numpy array in conversation...")
    conversation[1]["content"][0] = {"type": "audio", "audio": audio_np}
    try:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        print("SUCCESS with numpy in conversation")
    except Exception as e2:
        print(f"Failed: {e2}")

        # Approach 3: Inspect what processor.__call__ actually does
        print("\nDumping processor.__call__ source...")
        import inspect
        src = inspect.getsource(processor.__call__)
        # Find how 'audio' param is used
        for i, line in enumerate(src.split('\n')):
            if 'audio' in line.lower():
                print(f"  Line {i}: {line.strip()}")

        # Approach 4: Direct call with proper expansion
        print("\nTrying processor() directly...")
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        # Reset conversation
        conversation[1]["content"][0] = {"type": "audio", "audio": audio_path}

        inputs = processor(
            text=text,
            audio=[audio_np],
            return_tensors="pt",
            padding=True,
        )

print(f"\nInput keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}")

print(f"\n** input_ids size: {inputs['input_ids'].shape[1]} **")
if inputs['input_ids'].shape[1] < 1000:
    print("WARNING: Audio tokens NOT expanded! Model will not hear audio.")
else:
    print("OK: Audio tokens appear to be expanded.")

inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, 'to')}

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