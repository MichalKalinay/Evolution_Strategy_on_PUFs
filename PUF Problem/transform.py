# These are the copied transformation functions from the pypuf library
from typing import Callable

from numpy import ndarray, concatenate


def generate_stacked_transform(transform_1: Callable, k1: int, transform_2: Callable) -> Callable:
    """
    Combines input transformations ``transform_1`` and ``transform_2`` into a `stacked` input transformation,
    where the first ``k1`` sub-challenges are obtained by using ``transform_1`` on the challenge, and the remaining
    ``k - k1`` are sub-challenges are generated using ``transform_2``.
    """

    def transform(challenges: ndarray, k: int) -> ndarray:
        """Generates sub-challenges by applying different input transformations depending on the index
        of the sub-challenge."""
        (N, n) = challenges.shape
        transformed_1 = transform_1(challenges, k1)
        transformed_2 = transform_2(challenges, k - k1)
        assert transformed_1.shape == (N, k1, n)
        assert transformed_2.shape == (N, k - k1, n)
        return concatenate(
            (
                transformed_1,
                transformed_2,
            ),
            axis=1
        )

    transform.__name__ = f'transform_stack_{k1}_{transform_1.__name__.replace("transform_", "")}_' \
                         f'{transform_2.__name__.replace("transform_", "")}'

    return transform


def generate_concatenated_transform(transform_1: Callable, n1: int, transform_2: Callable) -> Callable:
    """
    Combines input transformations ``transform_1`` and ``transform_2`` into a `concatenated` input transformation,
    where the first ``n1`` bit of each sub-challenges are obtained by using ``transform_1`` on the first ``n1`` bit
    of the challenge, and the remaining ``n - n1`` bit of each sub-challenge are generated using ``transform_2``
    on the remaining ``n - n1`` bit of each given challenge.
    """

    def transform(challenges: ndarray, k: int) -> ndarray:
        """Generates sub-challenges by applying different input transformations depending on the index
        of the sub-challenge bit."""
        (N, n) = challenges.shape
        challenges1 = challenges[:, :n1]
        challenges2 = challenges[:, n1:]
        transformed_1 = transform_1(challenges1, k)
        transformed_2 = transform_2(challenges2, k)
        assert transformed_1.shape == (N, k, n1)
        assert transformed_2.shape == (N, k, n - n1)
        return concatenate(
            (
                transformed_1,
                transformed_2
            ),
            axis=2
        )

    transform.__name__ = f'transform_concat_{n1}_{transform_1.__name__.replace("transform_", "")}_' \
                         f'{transform_2.__name__.replace("transform_", "")}'

    return transform


def att(sub_challenges: ndarray) -> None:

    (_, _, n) = sub_challenges.shape
    for i in range(n - 2, -1, -1):
        sub_challenges[:, :, i] *= sub_challenges[:, :, i + 1]


def att_inverse(sub_challenges: ndarray) -> None:

    (_, _, n) = sub_challenges.shape
    for i in range(n - 1):
        sub_challenges[:, :, i] *= sub_challenges[:, :, i + 1]
