import pycuda.driver
import pycuda.autoinit
import torch

FORCE_CPU = False
use_gpu = False;
free, total = pycuda.driver.mem_get_info()
if FORCE_CPU:
    use_gpu = False;
elif torch.cuda.is_available() and free > 1500000000:
    device = torch.device("cuda")
    use_gpu = True;
else:
    use_gpu = False;

if use_gpu:
    device = torch.device("cuda")
    dtype = torch.float32
    print('USING GPU!')
else:
    device = torch.device("cpu")
    dtype = torch.float64
    print('using cpu :(')
