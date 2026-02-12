from safetensors.torch import load_file, save_file
import os

def fix_lora_keys(lora_dir, src_name="pytorch_lora_weights.safetensors", dst_name="pytorch_lora_weights_fixed.safetensors"):
    src_path = os.path.join(lora_dir, src_name)
    dst_path = os.path.join(lora_dir, dst_name)

    state = load_file(src_path)
    new_state = {}

    for k, v in state.items():
        # remove a leading "module." part if present
        # also handle "something.module.whatever" just in case
        parts = k.split(".")
        parts = [p for p in parts if p != "module"]
        new_key = ".".join(parts)
        new_state[new_key] = v

    save_file(new_state, dst_path)
    print(f"[fix_lora] wrote fixed weights to {dst_path}")
    return dst_path
