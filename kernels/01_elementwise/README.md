# element-wise add 优化

nsys 命令:
```bash
nsys profile                \
-t cuda,nvtx                \
-o {your_profile_filename}  \
--cuda-memory-usage true    \
--force-overwrite true       \
python3 elementwise.py
```

然后就会保存一个 {your_profile_filename}.nsys-rep 的文件, 可以在 nsight system 里面打开

ncu 命令:
```bash
sudo -E env PATH="$PATH"    \
VIRTUAL_ENV="$VIRTUAL_ENV"  \
PYTHONPATH="$PYTHONPATH"    \
/usr/local/cuda/bin/ncu     \
--set full                  \
-o ncu_elementwise          \
-f                          \
python elementwise.py
```

为了让它能够用 virtual env 下面的环境, 需要用 sudo -E env 设置一下
每次 kernel 都会做 ncu, 所以把 warmup 和 iter 调小一点
