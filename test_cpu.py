import torch
import time

# Define the size of the tensors
size = (20000, 20000)

# Create two tensors of the defined size
tensor1 = torch.randn(size)
tensor2 = torch.randn(size)

# Start the timer
start_time = time.time()

# Perform a matrix multiplication operation
result = torch.matmul(tensor1, tensor2)

# End the timer
end_time = time.time()

# Calculate the process time
process_time = end_time - start_time

# Print the result and the process time
print(result)
print(f"Process time: {process_time} seconds")