input_file = 'more/warmup/alllines.txt'
output_file = 'output.txt'

try:
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Remove both double quotes and single quotes
            modified_line = line.replace('"', '').replace("'", "")
            f_out.write(modified_line)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
