"""General Utilities for the black hole shadow code."""

class Metric:
    """Class to represent a metric."""
    def __init__(self, A, B, dAdr, dBdr):
        self.A = A
        self.B = B
        self.dAdr = dAdr
        self.dBdr = dBdr