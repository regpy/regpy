# from .utils.SVD_methods import randomized_SVD

# class GaussianApproximation(object):
#     def __init__(self, pdf):
#         # find m_MAP by Maximum-Likelihood
#         # TODO: Insert one of the implemented solvers instead of scipy.optimize.minimize
#         # Is done in mcmc_second_variant.
#         # Insert approximated code to compute gamma_prior_half^{1/2}
#
#         self.pdf = pdf
#         self.stepsize = 'randomly chosen'
#         self.y_MAP = self.pdf.setting.op(self.pdf.initial_state.positions)
#         N = self.pdf.initial_state.positions.shape[0]
#         # define the prior-preconditioned Hessian
#         self.Hessian_prior = np.zeros((N, N))
#         self.gamma_prior_inv = np.zeros((N, N))
#         for i in range(0, N):
#             self.gamma_prior_inv[:, i] = self.pdf.prior.hessian(self.pdf.m_0, np.eye(N)[:, i])
#         D, S = np.linalg.eig(-np.linalg.inv(self.gamma_prior_inv))
#         D = D.real
#         S = S.real
#
#         self.gamma_prior_half = np.dot(S.transpose(), np.dot(np.diag(np.sqrt(D)), S))
#         #
#         for i in range(0, N):
#             self.Hessian_prior[:, i] = np.dot(
#                 self.gamma_prior_half,
#                 self.pdf.likelihood.hessian(self.pdf.m_0, np.dot(self.gamma_prior_half, np.eye(N)[:, i]))
#             )
#         # construct randomized SVD of Hessian_prior
#         self.L, self.V = randomized_SVD(self, self.Hessian_prior)
#         self.L = -self.L
#         # define gamma_post
#         self.gamma_post = np.dot(
#             self.gamma_prior_half,
#             np.dot(
#                 self.V,
#                 np.dot(np.diag(1 / (self.L + 1)), np.dot(self.V.transpose(), self.gamma_prior_half))
#             )
#         )
#         self.gamma_post_half = np.dot(
#             self.gamma_prior_half,
#             np.dot(self.V, np.dot(np.diag(1 / np.sqrt(self.L + 1) - 1), self.V.transpose())) +
#             np.eye(self.gamma_prior_half.shape[0])
#         )
#
#     def random_samples(self):
#         R = np.random.normal(0, 1, self.gamma_post_half.shape[0])
#         m_post = self.pdf.initial_state.positions + np.dot(self.gamma_post_half, R)
#         return m_post
#
#     def next(self):
#         m_post = self.random_samples()
#         next_state = State()
#         next_state.positions = m_post
#         next_state.log_prob = np.exp(-np.dot(m_post, np.dot(self.gamma_post, m_post)))
#         self.state = next_state
#         return True
