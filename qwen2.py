import os
import inspect
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2-Audio-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

prompt = (
    "Describe the speaker's voice. Provide as many qualities of the voice as possible, "
    "and make them as detailed as possible. The overall description should allow someone "
    "to be able to identify that voice by listening to it and identifying the same features."
)

# Detect which processor call signature is in use (varies by transformers version)
sig = inspect.signature(processor.__call__)
param_names = list(sig.parameters.keys())

data_dir = "data"
output_base = "output"

for dataset in os.listdir(data_dir):
    dataset_path = os.path.join(data_dir, dataset)
    if not os.path.isdir(dataset_path):
        continue

    for filename in sorted(os.listdir(dataset_path)):
        if not filename.endswith(".wav"):
            continue

        audio_path = os.path.abspath(os.path.join(dataset_path, filename))
        stem = os.path.splitext(filename)[0]

        out_dir = os.path.join(output_base, dataset, stem)
        out_file = os.path.join(out_dir, "qwen2-audio.txt")

        if os.path.exists(out_file):
            print(f"[SKIP] {out_file} already exists")
            continue

        print(f"[PROCESSING] {dataset}/{filename}")

        # Load audio at 16 kHz
        audio, sr = librosa.load(audio_path, sr=16000)

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

        if "audios" in param_names:
            inputs = processor(
                text=[text],
                audios=[audio],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
        elif "audio" in param_names:
            inputs = processor(
                text=[text],
                audio=[audio],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
        else:
            print(f"Available params: {param_names}")
            print("Trying feature_extractor directly...")
            from transformers import AutoFeatureExtractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            audio_inputs = feature_extractor(
                [audio], sampling_rate=16000, return_tensors="pt"
            )
            text_inputs = processor.tokenizer(
                [text], return_tensors="pt", padding=True
            )
            inputs = {**text_inputs, **audio_inputs}

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        os.makedirs(out_dir, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(response)

        print(f"[SAVED] {out_file}")
        print("=" * 60)

print("\nDone! All files processed.")