import os
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor

repo_id = "mistralai/Voxtral-Small-24B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = (
    "Describe the speaker's voice. Provide as many qualities of the voice as possible, "
    "and make them as detailed as possible. The overall description should allow someone "
    "to be able to identify that voice by listening to it and identifying the same features."
)

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
        out_file = os.path.join(out_dir, "voxtral.txt")

        if os.path.exists(out_file):
            print(f"[SKIP] {out_file} already exists")
            continue

        print(f"[PROCESSING] {dataset}/{filename}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(model.device, dtype=torch.bfloat16)

        outputs = model.generate(**inputs, max_new_tokens=500)
        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        response = decoded_outputs[0]

        os.makedirs(out_dir, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(response)

        print(f"[SAVED] {out_file}")
        print("=" * 60)

print("\nDone! All files processed.")