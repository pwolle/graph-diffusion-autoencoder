import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bce_logits_1(binary, logits):
    l = 0
    l += binary * np.log(sigmoid(logits))
    l += (1 - binary) * np.log(1 - sigmoid(logits))
    return -l


def log_sigmoid(x):
    return -np.log(1 + np.exp(-x))


def bce_logits_2(binary, logits):
    l = 0
    l += binary * log_sigmoid(logits)
    l += (1 - binary) * log_sigmoid(-logits)
    return -l


def bce_logis_3(binary, logits):
    clipped = np.clip(logits, 0, None)
    return clipped - logits * binary + np.log(1 + np.exp(-np.abs(logits)))


def main():
    logits = np.random.randn(4)
    binary = np.random.randint(0, 2, 4)

    bce_1 = bce_logits_1(binary, logits)
    bce_2 = bce_logits_2(binary, logits)
    bce_3 = bce_logis_3(binary, logits)

    print(bce_1)
    print(bce_2)
    print(bce_3)

    assert np.allclose(bce_1, bce_2)
    assert np.allclose(bce_1, bce_3)
    assert np.allclose(bce_2, bce_3)


if __name__ == "__main__":
    main()
