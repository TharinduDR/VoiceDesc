import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

model_name = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

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
        out_file = os.path.join(out_dir, "qwen-omni.txt")

        if os.path.exists(out_file):
            print(f"[SKIP] {out_file} already exists")
            continue

        print(f"[PROCESSING] {dataset}/{filename}")

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

        # Build the text prompt from the conversation (no media kwargs here)
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Extract media from the conversation
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        # Call the processor with text + media together
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        inputs = inputs.to(model.device)

        # Match dtype for float tensors
        for k, v in inputs.items():
            if hasattr(v, "dtype") and v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                inputs[k] = v.to(model.dtype)

        output = model.generate(**inputs, max_new_tokens=256, return_audio=False)

        if isinstance(output, tuple):
            text_ids = output[0]
        else:
            text_ids = output

        generated_ids = text_ids[:, inputs["input_ids"].size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        os.makedirs(out_dir, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(response)

        print(f"[SAVED] {out_file}")
        print("=" * 60)

print("\nDone! All files processed.")