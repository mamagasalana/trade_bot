import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

import tensorflow as tf
import tensorflow_probability as tfp

# https://sarem-seitz.com/posts/multivariate-garch-with-python-and-tensorflow.html#using-python-and-tensorflow-to-implement-dcc
# credits to sarem seitz
class MGARCH_DCC(tf.keras.Model):
    """
    Tensorflow/Keras implementation of multivariate GARCH under dynamic conditional correlation (DCC) specification.
    Further reading:
        - Engle, Robert. "Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models."
        - Bollerslev, Tim. "Modeling the Coherence in Short-Run Nominal Exchange Rates: A Multi-variate Generalized ARCH Model."
        - Lütkepohl, Helmut. "New introduction to multiple time series analysis."
    """
    
    def __init__(self, y):
        """
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        super().__init__()
        n_dims = y.shape[1]
        self.n_dims = n_dims
        
        self.MU = tf.Variable(np.mean(y,0)) #use a mean variable
        
        self.sigma0 = tf.Variable(np.std(y,0)) #initial standard deviations at t=0
        
        #we initialize all restricted parameters to lie inside the desired range
        #by keeping the learning rate low, this should result in admissible results
        #for more complex models, this might not suffice
        self.alpha0 = tf.Variable(np.std(y,0))
        self.alpha = tf.Variable(tf.zeros(shape=(n_dims,))+0.25)
        self.beta = tf.Variable(tf.zeros(shape=(n_dims,))+0.25)
        
        self.L0 = tf.Variable(np.float32(np.linalg.cholesky(np.corrcoef(y.T)))) #decomposition of A_0
        self.A = tf.Variable(tf.zeros(shape=(1,))+0.9)
        self.B = tf.Variable(tf.zeros(shape=(1,))+0.05)
        
           
    def call(self, y):
        """
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        return self.get_conditional_dists(y)
    
    
    def get_log_probs(self, y):
        """
        Calculate log probabilities for a given matrix of time-series observations
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        return self.get_conditional_dists(y).log_prob(y)
    
        
    @tf.function
    def get_conditional_dists(self, y):
        """
        Calculate conditional distributions for given observations
        Args:
            y: NxM numpy.array of N observations of M correlated time-series
        """
        T = tf.shape(y)[0]
        
        #create containers for looping
        mus = tf.TensorArray(tf.float32, size = T) #observation mean container
        Sigmas = tf.TensorArray(tf.float32, size = T) #observation covariance container

        
        sigmas = tf.TensorArray(tf.float32, size = T+1)
        us = tf.TensorArray(tf.float32, size = T+1)
        Qs = tf.TensorArray(tf.float32, size = T+1)
        
        
        #initialize respective values for t=0
        sigmas = sigmas.write(0, self.sigma0)
        A0 = tf.transpose(self.L0)@self.L0
        Qs = Qs.write(0, A0) #set initial unnormalized correlation equal to mean matrix
        us = us.write(0, tf.zeros(shape=(self.n_dims,))) #initial observations equal to zero
        
        
        #convenience
        sigma0 = self.sigma0
        alpha0 = self.alpha0**2 #ensure positivity
        alpha = self.alpha
        beta = self.beta

        A = self.A
        B = self.B
        
        
        for t in tf.range(T):
            #tm1 = 't minus 1'
            #suppress conditioning on past in notation
            
            #1) calculate conditional standard deviations
            u_tm1 = us.read(t) 
            sigma_tm1 = sigmas.read(t)
            
            sigma_t = (alpha0 + alpha*sigma_tm1**2 + beta*u_tm1**2)**0.5
            
            #2) calculate conditional correlations
            u_tm1_standardized = u_tm1/sigma_tm1
                   
            Psi_tilde_tm1 = tf.reshape(u_tm1_standardized, (self.n_dims,1))@tf.reshape(u_tm1_standardized, (1,self.n_dims))

            Q_tm1 = Qs.read(t)
            Q_t = A0 + A*(Q_tm1 - A0) + B*(Psi_tilde_tm1 - A0)
            R_t = self.cov_to_corr(Q_t)
            
            #3) calculate conditional covariance
            D_t = tf.linalg.LinearOperatorDiag(sigma_t)
            Sigma_t = D_t@R_t@D_t
              
            
            #4) store values for next iteration
            sigmas = sigmas.write(t+1, sigma_t)
            us = us.write(t+1, y[t,:]-self.MU) #we want to model the zero-mean disturbances
            Qs = Qs.write(t+1, Q_t)
            
            mus = mus.write(t, self.MU)
            Sigmas = Sigmas.write(t, Sigma_t)
            
        return tfp.distributions.MultivariateNormalFullCovariance(mus.stack(), Sigmas.stack())
    
    
    def cov_to_corr(self, S):
        """
        Transforms covariance matrix to a correlation matrix via matrix operations
        Args:
            S: Symmetric, positive semidefinite covariance matrix (tf.Tensor)
        """
        D = tf.linalg.LinearOperatorDiag(1/(tf.linalg.diag_part(S)**0.5))
        return D@S@D
        
    

    def train_step(self, data):
        """
        Custom training step to handle keras model.fit given that there is no input-output structure in our model
        Args:
            S: Symmetric, positive semidefinite covariance matrix (tf.Tensor)
        """
        x,y = data
        with tf.GradientTape() as tape:
            loss = -tf.math.reduce_mean(self.get_log_probs(y))
            
        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {"Current loss": loss}
    
    
    
    def sample_forecast(self, y, T_forecast = 30, n_samples=500):
        """
        Create forecast samples to use for monte-carlo simulation of quantities of interest about the forecast (e.g. mean, var, corr, etc.)
        WARNING: This is not optimized very much and can take some time to run, probably due to Python's slow loops - can likely be improved
        Args:
            y: numpy.array of training data, used to initialize the forecast values
            T_forecast: number of periods to predict (integer)
            n_samples: Number of samples to draw (integer)
        """
        T = tf.shape(y)[0]
        
        #create lists for looping; no gradients, thus no tf.TensorArrays needed
        #can initialize directly
        mus = []
        Sigmas = []

        us = [tf.zeros(shape=(self.n_dims,))]
        sigmas = [self.sigma0]        
        Qs = []
        
        #initialize remaining values for t=0
        A0 = tf.transpose(self.L0)@self.L0
        Qs.append(A0)
        
        
        #convenience
        sigma0 = self.sigma0 
        alpha0 = self.alpha0**2 #ensure positivity
        alpha = self.alpha
        beta = self.beta

        A = self.A
        B = self.B
        
        #'warmup' to initialize latest lagged features
        for t in range(T):
            #tm1 = 't minus 1'
            #suppress conditioning on past in notation
            u_tm1 = us[-1]
            sigma_tm1 = sigmas[-1]
            
            sigma_t = (alpha0 + alpha*sigma_tm1**2 + beta*u_tm1**2)**0.5
            
            u_tm1_standardized = u_tm1/sigma_tm1
            
            Psi_tilde_tm1 = tf.reshape(u_tm1_standardized, (self.n_dims,1))@tf.reshape(u_tm1_standardized, (1,self.n_dims))

            Q_tm1 = Qs[-1]
            Q_t = A0 + A*(Q_tm1 - A0) + B*(Psi_tilde_tm1 - A0)
            R_t = self.cov_to_corr(Q_t)
            
            D_t = tf.linalg.LinearOperatorDiag(sigma_t)
            Sigma_t = D_t@R_t@D_t
              
            
            sigmas.append(sigma_t)
            us.append(y[t,:]-self.MU) #we want to model the zero-mean disturbances
            Qs.append(Q_t)
            
            mus.append(self.MU)
            Sigmas.append(Sigma_t)
  
            
        #sample containers
        y_samples = []
        R_samples = []
        sigma_samples = []
        
        
        for n in range(n_samples):
            
            mus_samp = []
            Sigmas_samp = []

            sigmas_samp = [sigmas[-1]]
            us_samp = [us[-1]]
            Qs_samp = [Qs[-1]]
            
            #forecast containers
            ys_samp = []
            sig_samp = []
            R_samp = [] 
            
            
            for t in range(T_forecast):
                u_tm1 = us_samp[-1]
                sigma_tm1 = sigmas_samp[-1]

                sigma_t = (alpha0 + alpha**2 + beta*u_tm1**2)**0.5

                u_tm1_standardized = u_tm1/sigma_tm1
                
                Psi_tilde_tm1 = tf.reshape(u_tm1_standardized, (self.n_dims,1))@tf.reshape(u_tm1_standardized, (1,self.n_dims))

                Q_tm1 = Qs_samp[-1]
                Q_t = A0 + A*(Q_tm1 - A0) + B*(Psi_tilde_tm1 - A0)
                R_t = self.cov_to_corr(Q_t)

                D_t = tf.linalg.LinearOperatorDiag(sigma_t)
                Sigma_t = D_t@R_t@D_t


                sigmas_samp.append(sigma_t)
                Qs_samp.append(Q_t)
                
                ynext = tfp.distributions.MultivariateNormalFullCovariance(self.MU, Sigma_t).sample()
                ys_samp.append(tf.reshape(ynext,(1,1,-1)))
                sig_samp.append(tf.reshape(sigma_t,(1,1,-1)))
                R_samp.append(tf.reshape(R_t,(1,1,self.n_dims,self.n_dims)))
                
                us_samp.append(ynext-self.MU)
                
            y_samples.append(tf.concat(ys_samp,1))
            R_samples.append(tf.concat(R_samp,1))
            sigma_samples.append(tf.concat(sig_samp,1))
        
        return tf.concat(y_samples,0).numpy(), tf.concat(R_samples,0).numpy(), tf.concat(sigma_samples,0).numpy()
    
class DCCGARCH_Multi:
    def __init__(self, returns, p=1, q=1, window=30):
        """
        Multivariate DCC-GARCH model.
        :param returns: DataFrame of asset returns.
        :param p: Order of the ARCH term.
        :param q: Order of the GARCH term.
        :param window: Rolling window size for correlation estimation (currently not used).
        """
        self.returns = returns
        self.p = p
        self.q = q
        self.window = window
        self.n_assets = returns.shape[1]
        self.garch_models = {}
        self.cond_vols = pd.DataFrame(index=returns.index, columns=returns.columns)
        self.std_resid = pd.DataFrame(index=returns.index, columns=returns.columns)
        self.R_bar = None
        self.DCC_params = None

    def fit_garch(self):
        """Fit univariate GARCH models to estimate conditional variances and compute standardized residuals."""
        for col in self.returns.columns:
            model = arch_model(self.returns[col], vol="Garch", p=self.p, q=self.q)
            fitted = model.fit(disp="off")
            self.garch_models[col] = fitted
            self.cond_vols[col] = fitted.conditional_volatility
            # Compute standardized residuals: residual / conditional volatility
            # Here we use the observed return as a proxy for the residual.
            self.std_resid[col] = self.returns[col] / self.cond_vols[col]

    def estimate_static_corr(self):
        """Estimate the unconditional correlation matrix R̄ using standardized residuals."""
        # Using the full sample; you could incorporate the 'window' here for rolling estimation.
        return self.std_resid.corr().values

    def dcc_likelihood(self, params):
        """Negative log-likelihood function for DCC-GARCH estimation using standardized residuals."""
        alpha, beta = params
        T, N = self.std_resid.shape
        Qt = np.zeros((T, N, N))
        Rt = np.zeros((T, N, N))
        
        # Estimate static correlation matrix R̄ from standardized residuals
        self.R_bar = self.estimate_static_corr()
        Qt[0] = self.R_bar  # Initialize Q_0

        log_likelihood = 0.0
        epsilon = 1e-6  # small constant for numerical stability

        # Loop over time, starting at t=1
        for t in range(1, T):
            # Use outer product of standardized residuals from previous time period
            u_tm1 = self.std_resid.iloc[t-1].values
            Qt[t] = (1 - alpha - beta) * self.R_bar + alpha * np.outer(u_tm1, u_tm1) + beta * Qt[t-1]
            
            # Compute R_t from Q_t
            diag_std = np.sqrt(np.diag(Qt[t]))
            Rt[t] = Qt[t] / np.outer(diag_std, diag_std)
            # Add a small constant to the diagonal for stability
            Rt[t] += epsilon * np.eye(N)

            # Invert R_t and compute its log-determinant
            try:
                inv_Rt = np.linalg.inv(Rt[t])
                sign, logdet = np.linalg.slogdet(Rt[t])
                if sign <= 0:
                    return np.inf
            except np.linalg.LinAlgError:
                return np.inf

            # Get standardized residual for time t
            u_t = self.std_resid.iloc[t].values
            quad_form = u_t.T @ inv_Rt @ u_t
            # Accumulate the likelihood contribution (with conventional 0.5 factor)
            log_likelihood += 0.5 * (logdet + quad_form)
            
        return log_likelihood  # Minimization target

    def fit_dcc(self):
        """Fit the DCC model by optimizing the likelihood function."""
        # Initial static correlation is estimated from standardized residuals
        self.R_bar = self.estimate_static_corr()
        bounds = [(0, 1), (0, 1)]  # Alpha and Beta must be between 0 and 1
        initial_params = [0.01, 0.98]  # Starting values
        result = minimize(self.dcc_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
        if not result.success:
            raise ValueError("DCC optimization failed: " + result.message)
        self.DCC_params = result.x  # Optimal α, β

    def fit(self):
        """Run the full Multivariate DCC-GARCH estimation."""
        print("Fitting univariate GARCH models and computing standardized residuals...")
        self.fit_garch()
        print("Estimating DCC parameters...")
        self.fit_dcc()
        print("DCC-GARCH model estimation complete.")

    def get_dcc_params(self):
        """Return estimated DCC parameters α, β."""
        return self.DCC_params

# Example Usage
if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(1000, 3), columns=["EURUSD", "GBPUSD", "USDJPY"])

    dcc_model = DCCGARCH_Multi(returns)
    dcc_model.fit()

    print("Estimated DCC Parameters (α, β):", dcc_model.get_dcc_params())
