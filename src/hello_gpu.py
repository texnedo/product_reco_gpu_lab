import subprocess
import time

import torch


print("=" * 70)
print("HELLO FROM AZURE ML GPU")
print("=" * 70)

print(f"PyTorch version:       {torch.__version__}")
print(f"CUDA available:        {torch.cuda.is_available()}")
print(f"PyTorch CUDA version:  {torch.version.cuda}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")

device = torch.device("cuda:0")

print(f"GPU count:             {torch.cuda.device_count()}")
print(f"GPU device:            {torch.cuda.get_device_name(0)}")

props = torch.cuda.get_device_properties(0)

print(f"GPU VRAM:              {props.total_memory / 1024**3:.2f} GB")
print(f"Compute capability:    {props.major}.{props.minor}")

print()
print("nvidia-smi:")
subprocess.run(["nvidia-smi"], check=False)

# ------------------------------------------------------------
# Tiny inference-like workload
# ------------------------------------------------------------

print()
print("=" * 70)
print("RUNNING SIMPLE GPU INFERENCE")
print("=" * 70)

torch.manual_seed(42)

model = torch.nn.Sequential(
    torch.nn.Linear(1024, 4096),
    torch.nn.GELU(),
    torch.nn.Linear(4096, 1024),
).to(device).half()

model.eval()

batch = torch.randn(
    128,
    1024,
    device=device,
    dtype=torch.float16,
)

# Warmup.
with torch.no_grad():
    for _ in range(20):
        _ = model(batch)

torch.cuda.synchronize()

iterations = 100

start = time.perf_counter()

with torch.no_grad():
    for _ in range(iterations):
        output = model(batch)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

latency_ms = elapsed / iterations * 1000
samples_per_second = batch.shape[0] * iterations / elapsed

print(f"Iterations:            {iterations}")
print(f"Batch size:            {batch.shape[0]}")
print(f"Average latency:       {latency_ms:.3f} ms")
print(f"Throughput:            {samples_per_second:,.0f} samples/sec")
print(f"Output shape:          {tuple(output.shape)}")

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2

print(f"CUDA allocated memory: {allocated:.1f} MB")
print(f"CUDA reserved memory:  {reserved:.1f} MB")

print()
print("SUCCESS: GPU inference completed.")