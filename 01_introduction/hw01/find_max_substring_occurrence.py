def find_max_substring_occurrence(input_string):
    for idx in range(1, len(input_string) + 1):
        if input_string[:idx] * (len(input_string) // idx) == input_string:
            return len(input_string) // idx
