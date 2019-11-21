from collections import defaultdict


def get_new_dictionary(input_dict_name, output_dict_name):
    with open(input_dict_name, 'r') as f:
        f.readline()
        result = defaultdict(list)
        for line in f:
            base, other = line.rstrip('\n').split(' - ')
            for word in other.split(', '):
                result[word].append(base)

    with open(output_dict_name, 'w') as f:
        f.write(str(len(result)) + '\n')
        for word in sorted(result.keys()):
            f.write(word + ' - ' + ', '.join(sorted(result[word])) + '\n')
