import torch
import time

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.set_default_device('mps')
#device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print('set device to mps')
# Define the size of the tensors
n = 32000
print(f'n={n}')
size = (n, n)
torch.set_num_threads(torch.get_num_threads())
print('size set')
# Create two tensors of the defined size and move them to the GPU
tensor1 = torch.randn(size, device=device)
tensor2 = torch.randn(size, device=device)
print('tensors set')
# Start the timer
try:
    start_time = time.time()
    print(f'start at {start_time}')
    # Perform a matrix multiplication operation
    result = torch.matmul(tensor1, tensor2)

    # End the timer
    end_time = time.time()

    # Calculate the process time
    process_time = end_time - start_time

    # Print the result and the process time
    print(f"Process time: {process_time} seconds")
    print(result[0,0].item())
    #print(torch.max(result))
except Exception as e:
    print(f"Error printing result: {e}")