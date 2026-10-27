"""Load OXR-1.0 on CPU with Transformers (no vLLM) and run one recognition request.

Usage: python oxr_cpu.py IMAGE [--prompt "Recognize the content of the image."]
"""

import argparse
import sys
import time

import torch
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_DIR = r"D:\HMIS_Ihl\OXR\models\OXR-1.0"
# Same pixel limits as the official vLLM plugin (oxr_vllm/oxr_processor.py).
MIN_PIXELS = 448 * 448
MAX_PIXELS = 960 * 960
EOS_IDS = [151643, 151645]


def load_model():
    # AutoModelForCausalLM imports configuration_oxr twice under different module
    # hashes, so its Siglip2NavitConfig doesn't match the class registered with
    # AutoModel. Build the config from the modeling module's own copy instead.
    oxr_cls = get_class_from_dynamic_module("modeling_oxr.OXR", MODEL_DIR)
    config_cls = sys.modules[oxr_cls.__module__].OXRConfig
    config = config_cls.from_pretrained(MODEL_DIR)
    model = oxr_cls.from_pretrained(MODEL_DIR, config=config, dtype=torch.float32)
    # Non-persistent buffer isn't in the checkpoint and is left uninitialised by
    # Transformers 5 meta-device loading; restore the value set in OXR.__init__.
    vocab = config.visual_vocab_size
    model.indicator_token_indices = torch.arange(vocab - 4, vocab, dtype=torch.long)
    return model.eval()


@torch.inference_mode()
def recognize(model, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> tuple[str, int]:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    output = model.generate(
        inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
        max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[0],
    )
    tokens = output[0]
    return model.text_tokenizer.decode(tokens, skip_special_tokens=True).strip(), len(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--prompt", default="Recognize the content of the image.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    started = time.perf_counter()
    model = load_model()
    print(f"Model loaded in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    text, n = recognize(model, Image.open(args.image), args.prompt, args.max_new_tokens)
    print(f"Generated {n} tokens in {time.perf_counter() - started:.1f}s\n---")
    print(text)


if __name__ == "__main__":
    main()
