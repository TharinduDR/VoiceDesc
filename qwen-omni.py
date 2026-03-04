from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model_name = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

audio_path = "my_speech.wav"

prompt = (
    "Describe the speaker's voice. Provide as many qualities of the voice as possible, and make them as detailed as possible. The overall description should allow someone to be able to identify that voice by listening to it and identifying the same features."
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

# New approach: apply_chat_template handles EVERYTHING
inputs = processor.apply_chat_template(
    conversation,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
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