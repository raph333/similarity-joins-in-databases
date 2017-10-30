import csv


def read_txt(filename):
    """
    reads in txt file from working directory
    :param filename: string *.txt
    :return: list of lists where each line from the txt file represents an integer list
    """
    result = []

    with open(filename) as input_file:
        for row in csv.reader(input_file):
            row = list(map(int, row))
            result.append(row)

    return result
