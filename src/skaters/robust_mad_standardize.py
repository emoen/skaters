import math

class RobustMADStandardize:
    """
    Streaming Robust Median and MAD Standardizer.
    Approximates tracking the rolling median and MAD in O(1) time per observation
    using stochastic gradient tracking (stochastic approximation).
    """
    def __init__(self, lr_median=0.01, lr_mad=0.01, initial_guess=0.0):
        self.median = initial_guess
        self.mad = 1.0  # Avoid zero-division at initialization
        self.lr_median = lr_median
        self.lr_mad = lr_mad
        
    def update(self, y: float) -> float:
        """
        Updates the running robust estimates with a new observation y.
        Returns the standardized value (y - median) / MAD.
        """
        # 1. Update the median using the sign function (stochastic quantile estimation)
        if y > self.median:
            self.median += self.lr_median * self.mad
        elif y < self.median:
            self.median -= self.lr_median * self.mad
            
        # 2. Compute the absolute deviation from the current median
        dev = abs(y - self.median)
        
        # 3. Update the MAD using sign-based tracking for the median deviation
        # 1.4826 converts MAD to an unbiased estimator of standard deviation under Normal distribution
        target_mad = dev / 1.4826
        
        if target_mad > self.mad:
            self.mad += self.lr_mad * self.mad
        elif target_mad < self.mad:
            self.mad -= self.lr_mad * self.mad
            
        # Bound MAD away from true zero to preserve structural stability
        if self.mad < 1e-9:
            self.mad = 1e-9
            
        # 4. Return the robustly studentized coordinate
        return (y - self.median) / self.mad

    def inverse(self, y_prime: float) -> float:
        """ Maps the standardized coordinate back to the original space. """
        return (y_prime * self.mad) + self.median
