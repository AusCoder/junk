import numpy as np


if __name__ == "__main__":
    cost_matrix = 1 - np.array(
        [
            [0.95, 0.76, 0.62, 0.41, 0.06],
            [0.23, 0.46, 0.79, 0.94, 0.35],
            [0.61, 0.02, 0.92, 0.92, 0.81],
            [0.49, 0.82, 0.74, 0.41, 0.01],
            [0.89, 0.44, 0.18, 0.89, 0.14],
        ]
    )

    print(cost_matrix)
