import tensorflow as tf
import numpy as np

class TF_kmeans:
    def __init__(self, K, max_iters = 100, tol = 1e-4):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol

    def initialize_centroids(self, X):
        indices = tf.random.shuffle(tf.range(len(X)))[:self.K]
        return tf.gather(X, indices)
    
    @tf.function
    def fit(self, X):
        # --- initialize ---
        centroids = self.initialize_centroids(X)

        for i in tf.range(self.max_iters):
            # E-step
            # (N, 1, D) - (1, K, D) -boardcast-> (N, K, D)
            dist = tf.norm(tf.expand_dims(X,1) - tf.expand_dims(centroids,0), axis = 2)

            assignments = tf.argmin(dist, axis = 1)

            # M-step
            new_centroids = tf.math.unsorted_segment_mean(
                data = X, segment_ids = assignments, num_segments = self.K
            )

            shift = tf.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break
        return centroids, assignments
    
    @tf.function
    def gmm_init_params(self, X):
        mu_init, assignments = self.fit(X)

        # Pi
        counts = tf.math.bincount(assignments, minlength = self.K)
        pi_init = tf.cast(counts, tf.float32) / tf.cast(X.shape[0], tf.float32)
        
        #Sigma
        mask = tf.one_hot(assignments, depth=self.K, dtype=tf.float32) # (N, K)
        covs = tf.TensorArray(tf.float32, size = self.K)
        for k in tf.range(self.K):
            diff = X - mu_init[k] # shape (N,D)
            weighted_diff_T = tf.transpose(diff) * mask[:, k]
            sigma_k = tf.matmul(weighted_diff_T, diff) / tf.maximum(tf.cast(counts[k], tf.float32), 1.0) #(N*D) * (D*N) -> (N*N)
            sigma_k = sigma_k + tf.eye(X.shape[1]) * 1e-6 # prevent exotic matrix
            covs = covs.write(k , sigma_k)
        sigma_init = covs.stack()
        
        return mu_init, pi_init, sigma_init
    
class TF_GMM:
    def __init__(self, K, max_iters = 100, tol = 1e-4):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        # Claim parameters
        self.mu = None
        self.sigma = None
        self.pi = None
    
    def initial_params(self, mu_init, pi_init, sigma_init):
        self.mu = tf.Variable(mu_init, dtype=tf.float32)
        self.pi = tf.Variable(pi_init, dtype=tf.float32)
        self.sigma = tf.Variable(sigma_init, dtype=tf.float32) 
    
    @tf.function
    def log_gaussian_pdf(self, X, Mu, Sigma):
        D = tf.cast(X.shape[1], tf.float32)

        diff = X - Mu
        L = tf.linalg.cholesky(Sigma)
        y = tf.linalg.triangular_solve(L, tf.transpose(diff), lower = True)
        dist_sq = tf.reduce_sum(tf.square(y), axis = 0)

        log_det_sigma = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        
        log_const = D * tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32))
        log_prob = -0.5 * (log_const + log_det_sigma + dist_sq)
        
        return log_prob
    
    @tf.function
    def E_step(self, X):
        # Calculate the resposiblity matrix (N,K)
        N = X.shape[0]

        log_prob_lists = tf.TensorArray(tf.float32, size = self.K)
        for k in range(self.K):
            log_prob = self.log_gaussian_pdf(X, self.mu[k], self.sigma[k])
            log_prob_lists = log_prob_lists.write(k, log_prob)

        log_pdfs = tf.transpose(log_prob_lists.stack())# list with k (N,) -> array (N,k)

        log_weighted_probs = log_pdfs + tf.math.log(self.pi + 1e-10)

        log_evidence = tf.math.reduce_logsumexp(log_weighted_probs, axis = 1, keepdims = True) # -> (N,1)

        log_gamma = log_weighted_probs - log_evidence
        gamma = tf.exp(log_gamma)

        return gamma, tf.reduce_sum(log_evidence)

    @tf.function
    def M_step(self, X, gamma):
        # gamma: (N, K) matrix
        N = tf.cast(tf.shape(X)[0], tf.float32)
        D = tf.shape(X)[1]

        # Nk (K,)
        Nk = tf.reduce_sum(gamma, axis = 0)

        # pi (K,)
        self.pi.assign(Nk / N)

        # mu (K, D)
        weight_matrix = tf.transpose(gamma) / tf.expand_dims(Nk, 1)
        self.mu.assign(tf.matmul(weight_matrix, X))

        # sigma (K, D, D)
        covs = tf.TensorArray(tf.float32, size = self.K)
        for k in range(self.K):
            diff = X - self.mu[k] #(N, D)
            weigth_diff = tf.transpose(diff) * gamma[:,k]
            sigma_k = tf.matmul(weigth_diff, diff) / Nk[k]
            sigma_k = sigma_k + tf.eye(D) * 1e-6
            covs = covs.write(k , sigma_k)
        self.sigma.assign(covs.stack())
    
    def fit(self, X, mu_init, pi_init, sigma_init, verbose = True):
        '''
        X: Training Data(N,D)
        mu_init, pi_init, sigma_init: initial parameters calculated from K-means
        verbose: print the training progress
        '''
        # Get initial parameters
        self.initial_params(mu_init, pi_init, sigma_init)

        # log likelihood, initial at -inf
        prev_log_likelihood = tf.constant(-np.inf, dtype=tf.float32)

        # final result gamma
        final_gamma = None

        if verbose:
            print('---GMM training---')
        
        for i in range(self.max_iters):
            gamma, current_log_likelihood = self.E_step(X)
            self.M_step(X, gamma)
            likelihood_shift = tf.abs(current_log_likelihood - prev_log_likelihood)
            
            if verbose and (i%5 == 0 or i == 0):
                print(f'Iteration: {i:3d} | Log_likelihood: {current_log_likelihood.numpy():.4f} | Shift: {likelihood_shift.numpy():.6f}')
            
            if likelihood_shift < self.tol:
                if verbose:
                    print(f'---Iteration end at {i}.---')
                final_gamma = gamma
                break
            
            prev_log_likelihood = current_log_likelihood
            final_gamma = gamma
        
        # calculate AIC BIC
        N = tf.cast(tf.shape(X)[0], tf.float32)
        D = tf.cast(tf.shape(X)[1], tf.float32)
        K_float = tf.cast(self.K, tf.float32)
        
        # total number of parameter p
        params_per_cluster = D + D * (D + 1.0) / 2.0
        total_params = K_float * params_per_cluster + (K_float - 1.0)
        
        # AIC and BIC
        aic = 2.0 * total_params - 2.0 * prev_log_likelihood
        bic = total_params * tf.math.log(N) - 2.0 * prev_log_likelihood
        
        if verbose:
            print(f"\n--- Model Metrics ---")
            print(f"Total Params : {total_params.numpy():.0f}")
            print(f"Final Log-L  : {prev_log_likelihood.numpy():.4f}")
            print(f"AIC          : {aic.numpy():.4f}")
            print(f"BIC          : {bic.numpy():.4f}")
        return final_gamma, prev_log_likelihood.numpy(), aic.numpy(), bic.numpy()

# Validation GMM function

@tf.function(reduce_retracing=True)
def log_gaussian_pdf(X, Mu, Sigma):
    D = tf.cast(X.shape[1], tf.float32)

    diff = X - Mu
    L = tf.linalg.cholesky(Sigma)
    y = tf.linalg.triangular_solve(L, tf.transpose(diff), lower = True)
    dist_sq = tf.reduce_sum(tf.square(y), axis = 0)

    log_det_sigma = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        
    log_const = D * tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32))
    log_prob = -0.5 * (log_const + log_det_sigma + dist_sq)
        
    return log_prob

@tf.function(reduce_retracing=True)
def GMM_E_Step(X, Mu, Sigma, Pi):
    K = tf.shape(Mu)[0]

    log_prob_lists = tf.TensorArray(tf.float32, size = K)
    for k in range(K):
        log_prob = log_gaussian_pdf(X, Mu[k], Sigma[k])
        log_prob_lists = log_prob_lists.write(k, log_prob)

    log_pdfs = tf.transpose(log_prob_lists.stack())# list with k (N,) -> array (N,k)

    log_weighted_probs = log_pdfs + tf.math.log(Pi + 1e-10)

    log_evidence = tf.math.reduce_logsumexp(log_weighted_probs, axis = 1, keepdims = True) # -> (N,1)

    log_gamma = log_weighted_probs - log_evidence
    gamma = tf.exp(log_gamma)

    return gamma, tf.reduce_sum(log_evidence)