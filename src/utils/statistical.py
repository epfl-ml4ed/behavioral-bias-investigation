
import numpy as np
from tqdm import tqdm


def fischer_mean(sample_a, sample_b, runs=10000):
    """Runs the 'mean'-fisherian random invariance test.

    Computes the difference of means between group A and group B.
    Then randomly shuffles who belongs to A and who belongs to B according to 
    these criteria during the bootstrap run:
    - the number of samples attributed to A/B need to be equal to the number of
    samples in A/B
    - all samples need to be attributed to either A or B, but not both of them, 
    and not neither of them

    Args:
        sample_a (list): samples belonging to group A
        sample_b (list): samples belonging to group B
        runs (int, optional): Number of bootstrapping. Defaults to 10000.
    """
    na = len(sample_a)
    nb = len(sample_b)
    n = na + nb
    sample_indices = [i for i in range(n)]
    all_samples = [*sample_a, *sample_b]
    original_mean_a = np.mean(sample_a)
    original_mean_b = np.mean(sample_b)
    original_mean_diff = np.abs(original_mean_a - original_mean_b)

    sup_means = 0
    for _ in (range(runs)):
        np.random.shuffle(all_samples)
        bootstrap_a = all_samples[:na]
        bootstrap_b = all_samples[na:]
        # print(bootstrap_a)
        assert len(bootstrap_a) == na and len(bootstrap_b) == nb
        # bootstrap_indices_a = [
        #     c for c in np.random.choice(sample_indices, size=na, replace=False)
        # ]
        # bootstrap_indices_b = list(set(sample_indices).difference(set(bootstrap_indices_a)))
        # bootstrap_indices_b = [
        #     b for b in range(n) if b not in bootstrap_indices_a
        # ]
        # bootstrap_a = [all_samples[idx] for idx in bootstrap_indices_a]
        # bootstrap_b = [all_samples[idx] for idx in bootstrap_indices_b]

        # assert len(bootstrap_a) == na and len(bootstrap_b) == nb
        # assert len(set(bootstrap_indices_b).intersection(set(bootstrap_indices_a))) == 0

        # print(bootstrap_b)
        # print(bootstrap_a)
        bootrstrap_mean_diff = np.abs(np.mean(bootstrap_a) - np.mean(bootstrap_b))
        # print(np.mean(bootstrap_a))
        # print(np.mean(bootstrap_b))
        # print(bootrstrap_mean_diff, original_mean_diff)
        if bootrstrap_mean_diff >= original_mean_diff:
            sup_means += 1
    fischer = sup_means / runs
    # print('Fischer test Value', fischer)
    return fischer

