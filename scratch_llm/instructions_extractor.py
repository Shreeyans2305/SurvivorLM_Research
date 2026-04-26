# Example using E2B/Novita style Python
import fs
import e2b_code_interpreter

# Assuming 'e2b_code_interpreter' is set up
# Download file from sandbox
file_content = fs.read('sandbox:/mnt/data/SurvivorLLM_Dataset_Foundational.txt')

# Write to local filesystem
with open('instructions_1.txt', 'wb') as f:
    f.write(file_content)