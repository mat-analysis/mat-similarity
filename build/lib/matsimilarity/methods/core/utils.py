import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import gen_even_slices


def similarity_matrix(T, measure=None, n_jobs=1):
    """
    Computes the similarity matrix from a list of trajectories T.

    Parameters:
    -----------
    T : list
        List of Trajectory objects. Each trajectory should be a MultipleAspectSequence.
    measure : SimilarityMeasure instance
        A class with a similarity function that takes two trajectories and returns a similarity score.
    n_jobs : int, optional
        The number of parallel jobs to use for computation. If -1, all processors 
        are used (default is 1).
    
    Returns:
    --------
    np.ndarray
        A 2D numpy array containing similarity scores between trajectories. 
        The element at [i, j] represents the similarity between trajectory T[i] and T[j].
    
    Example:
    --------
    >>> T = [Trajectory1, Trajectory2, Trajectory3]
    >>> sim_matrix = similarity_matrix(T, measure=MUITAS(), n_jobs=4)
    >>> print(sim_matrix)
    [[1.0, 0.8, 0.3],
     [0.8, 1.0, 0.5],
     [0.3, 0.5, 1.0]]
    """
    def process(X, s):
        matrix = np.zeros(shape=(len(X), len(X)))

        for i in range(s.start + 1, len(X)):
            for j in range(0, min(len(X), i - s.start)):
                matrix[i][j] = measure.similarity(X[i], X[j])
        return matrix

    func = delayed(process)

    similarity = Parallel(n_jobs=n_jobs, verbose=0)(
        func(X, X[s], s) for s in gen_even_slices(len(X), n_jobs))
    similarity = np.hstack(similarity)

    similarity += similarity.transpose() + np.identity(len(X))

    return similarity