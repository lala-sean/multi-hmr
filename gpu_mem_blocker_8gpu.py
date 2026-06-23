import argparse
import time

import torch


def parse_devices(raw):
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def get_free_bytes(dev_id):
    try:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.free
    except ModuleNotFoundError:
        free_bytes, _ = torch.cuda.mem_get_info(dev_id)
        return free_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--claim-fraction", type=float, default=0.95)
    parser.add_argument("--sleep-interval", type=float, default=0.05)
    args = parser.parse_args()

    target_devices = parse_devices(args.devices)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    try:
        import pynvml

        pynvml.nvmlInit()
    except ModuleNotFoundError:
        print("[i] pynvml is not installed; using torch.cuda.mem_get_info instead.", flush=True)

    tensors = []

    for dev_id in target_devices:
        free_bytes = get_free_bytes(dev_id)
        claim_bytes = int(free_bytes * args.claim_fraction)
        num_elements = claim_bytes // 4
        device = torch.device(f"cuda:{dev_id}")

        try:
            tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
            tensor.fill_(1.0)
            torch.cuda.synchronize(device)
            tensors.append((device, tensor))
            print(f"[ok] Blocked ~{claim_bytes / 1e9:.2f} GB on cuda:{dev_id}", flush=True)
        except RuntimeError as exc:
            print(f"[x] Failed to allocate on cuda:{dev_id}: {exc}", flush=True)
            raise

    print("\n[Running light GPU load; Ctrl+C to stop]\n", flush=True)

    try:
        while True:
            for device, tensor in tensors:
                with torch.cuda.device(device):
                    tensor.add_(1e-7)
            time.sleep(args.sleep_interval)
    except KeyboardInterrupt:
        print("Interrupted. Releasing GPU memory.", flush=True)


if __name__ == "__main__":
    main()
