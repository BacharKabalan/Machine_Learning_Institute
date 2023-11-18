def load_corpus(path = 'corpus.txt'):
    corpus = []

    # Specify the path to your text file
    file_path = path

    # Open and read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove leading and trailing whitespace and append the line to the list
            corpus.append(line.strip())

    # Now, text_lines contains each line of the text file as an element
    return corpus