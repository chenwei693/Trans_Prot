import torch
import pynvml

print('GPU is_available', torch.cuda.is_available())

pynvml.nvmlInit()
gpu_num = pynvml.nvmlDeviceGetCount()
print('gpu num:', gpu_num)  # 显示有几块GPU

for i in range(gpu_num):
    print('-' * 50, 'gpu[{}]'.format(str(i)), '-' * 50)
    gpu = pynvml.nvmlDeviceGetHandleByIndex(i)

    print('gpu object:', gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu)
    print('total memory:', meminfo.total / 1024 ** 3, 'GB')  # 第i块显卡总的显存大小
    print('using memory:', meminfo.used / 1024 ** 3, 'GB')  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    print('remaining memory:', meminfo.free / 1024 ** 3, 'GB')  # 第二块显卡剩余显存大小
print(torch.cuda.is_available())  # True 表示有 GPU 可用
print(torch.cuda.device_count())  # 可用 GPU 数量
print(torch.cuda.get_device_name(0))  # 获取 GPU 名称