import os
import torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity

# torch.set_grad_enabled(False)

# def set_cuda_arch() -> None:
#     major, minor = torch.cuda.get_device_capability()
#     if "TORCH_CUDA_ARCH_LIST" not in os.environ:
#         os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
#         print(f"[INFO] Set TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
#     else:
#         assert os.environ["TORCH_CUDA_ARCH_LIST"] == f"{major}.{minor}"

# set_cuda_arch()

lib = load(
    name="startup_lib",
    sources=[
        "startup.cu",
    ],
    extra_cuda_cflags=[
        "-O3",
    ],
    extra_cflags=["-std=c++17"],
)

def main() -> None:
    lib.hello_gpu()
    torch.cuda.synchronize()


def main_with_profiler(trace_path: str = "trace.json") -> None:
    """Run the same workload under torch.profiler and export a Chrome trace.

    Enable by setting environment variable `EXPORT_TRACE=1` when running the script.
    The produced `trace.json` can be opened with Chrome tracing (chrome://tracing)
    or uploaded to Perfetto UI (https://ui.perfetto.dev/).
    """
    # helper for NVTX context that works across torch versions
    def nvtx_range(name: str):
        nvtx = torch.cuda.nvtx
        if hasattr(nvtx, "range"):
            return nvtx.range(name)

        class _Range:
            def __enter__(self):
                nvtx.range_push(name)

            def __exit__(self, exc_type, exc, tb):
                nvtx.range_pop()

        return _Range()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False, profile_memory=True, with_stack=True) as prof:
        # put a high-level recorded scope around the GPU call
        with record_function("startup.hello_gpu"):
            with nvtx_range("hello_gpu"):
                lib.hello_gpu()
                torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"[INFO] Exported chrome trace to {trace_path}")

if __name__ == "__main__":
    if os.getenv("EXPORT_TRACE", "0") == "1":
        main_with_profiler("trace.json")
    else:
        main()
