from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

model_name = "Qwen/Qwen2-Audio-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

# Load audio at 16 kHz (required)
audio_path = "my_speech.wav"
audio, sr = librosa.load(audio_path, sr=16000)

prompt = (
    "Describe this speech in detail, including the language spoken, "
    "speaker characteristics, emotion, and content."
)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False
)
inputs = processor(
    text=[text],
    audios=[audio],
    sampling_rate=16000,
    return_tensors="pt",
    padding=True,
)
inputs = inputs.to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)