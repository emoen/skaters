class ResidualVarianceLens:
    """
    Tracks the running variance ratio between a recent online window
    and a longer historical baseline window to capture structural surprise.
    """
    def __init__(self, alpha_short=0.05, alpha_long=0.005):
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        
        # Track running means and variances using Welford/EMA hybrids
        self.mean_short = 0.0
        self.var_short = 1.0
        
        self.mean_long = 0.0
        self.var_long = 1.0
        
    def update(self, epsilon: float) -> float:
        """
        Takes a raw residual (epsilon) from a prior layer (e.g., an AR or EMA output).
        Returns the log variance ratio: log((var_short + eps) / (var_long + eps)).
        """
        # Update short-term variance tracker
        diff_short = epsilon - self.mean_short
        self.mean_short += self.alpha_short * diff_short
        self.var_short = (1.0 - self.alpha_short) * (self.var_short + self.alpha_short * diff_short * (epsilon - self.mean_short))
        
        # Update long-term baseline variance tracker
        diff_long = epsilon - self.mean_long
        self.mean_long += self.alpha_long * diff_long
        self.var_long = (1.0 - self.alpha_long) * (self.var_long + self.alpha_long * diff_long * (epsilon - self.mean_long))
        
        # Guard rails to prevent negative or zero scale collapses
        v_short = max(self.var_short, 1e-9)
        v_long = max(self.var_long, 1e-9)
        
        # Emit the log variance ratio feature (model-surprise metric)
        # Adds epsilon padding inside log to maintain numerical stability
        return math.log(v_short / v_long)

    def inverse(self, y_prime: float) -> float:
        """
        Since a lens mapping variance surprise alters the information space 
        rather than simple linear values, it passes the raw structural feature forward.
        """
        return y_prime
