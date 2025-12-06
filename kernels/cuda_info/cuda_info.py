from numba import cuda

def print_device_properties():
    device_count = len(cuda.gpus)
    print(f"CUDA Device Count: {device_count}")

    for i in range(device_count):
        dev = cuda.select_device(i)

        print(f"\n===== Device {i} =====")
        print("Name:", dev.name)

        # 获取所有属性字典
        attrs = dev.attributes

        print(f"Found {len(attrs)} attributes:\n")
        for k, v in attrs.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    print_device_properties()