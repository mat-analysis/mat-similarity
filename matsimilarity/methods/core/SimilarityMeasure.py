from abc import ABC, abstractmethod

class SimilarityMeasure(ABC):

    def similarity(self, t1: MultipleAspectSequence, t2: MultipleAspectSequence) -> float:
        """Computes the similarity score of the given MAT.

        Parameters
        ----------
        t1 : MultipleAspectSequence instance of the trajectory 1.
        t2 : MultipleAspectSequence instance of the trajectory 2.

        Returns
        -------
        score : float
            Similarity score (between 0 and 1).
        """
        pass