import csv


def read_txt(filename):
    """
    reads in txt file from working directory
    :param filename: string *.txt
    :return: list where each line from the txt file represents a list itself
    """
    result = []

    with open(filename) as input_file:
        for row in csv.reader(input_file):
            row = list(map(int, row))
            result.append(row)

    return result
