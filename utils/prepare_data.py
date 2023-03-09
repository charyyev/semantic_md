import numpy as np

def write_to_file(filename, data):
    with open(filename, 'a') as f:
        f.write(str(data))
        f.write('\n')


if __name__ == "__main__":
    num_data = 1449
    train_percent = 0.8
    val_percent = 0.1

    num_train = int(num_data * train_percent)
    num_val = int(num_data * val_percent)
    idxs = np.arange(num_data)
    np.random.shuffle(idxs)

    train_idxs = idxs[:num_train]
    val_idxs = idxs[num_train: num_train + num_val]
    test_idxs = idxs[num_train + num_val:]

    train_filename = "/home/sapar/3dvision/data/list/train.txt"
    val_filename = "/home/sapar/3dvision/data/list/val.txt"
    test_filename = "/home/sapar/3dvision/data/list/test.txt"

    for idx in train_idxs:
        write_to_file(train_filename, idx)
    for idx in val_idxs:
        write_to_file(val_filename, idx)
    for idx in test_idxs:
        write_to_file(test_filename, idx)
    