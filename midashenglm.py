import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

model_id = "mispeech/midashenglm-7b-0804-bf16"

with torch.device("cpu"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

model = model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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
        out_file = os.path.join(out_dir, "midashenglm.txt")

        if os.path.exists(out_file):
            print(f"[SKIP] {out_file} already exists")
            continue

        print(f"[PROCESSING] {dataset}/{filename}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        os.makedirs(out_dir, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(response)

        print(f"[SAVED] {out_file}")
        print("=" * 60)

print("\nDone! All files processed.")