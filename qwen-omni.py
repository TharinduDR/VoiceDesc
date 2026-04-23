import os
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Captioner"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

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
        out_file = os.path.join(out_dir, "qwen3-omni-captioner.txt")

        if os.path.exists(out_file):
            print(f"[SKIP] {out_file} already exists")
            continue

        print(f"[PROCESSING] {dataset}/{filename}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference
        text_ids, audio = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
        )

        response = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        os.makedirs(out_dir, exist_ok=True)
        with open(out_file, "w") as f:
            f.write(response)

        print(f"[SAVED] {out_file}")
        print("=" * 60)

print("\nDone! All files processed.")