from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

model_name = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

audio_path = "my_speech.wav"
prompt = (
    "Describe this speech in detail, including the language spoken, "
    "speaker characteristics, emotion, and content."
)

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant that analyzes audio clips.",
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
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

inputs = processor(
    text=text, audios=audios, images=images, videos=videos,
    return_tensors="pt", padding=True,
)
inputs = inputs.to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=256, use_audio_in_video=True)
generated_ids = output_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)