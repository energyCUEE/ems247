import numpy as np
import cvxpy as cp
def project_constant_block(C, n1:int):
  dimC = C.shape[0]
  if dimC % n1 != 0 :
    print("Error: the dimension of C is not a multiple of n1, can't split the blocks")
  else:
    n_blocks = int(dimC//n1)

  diag_blocks = np.array( [C[i*n1:(i+1)*n1,i*n1:(i+1)*n1] for i in range(n_blocks)] ) # n_blocks x n1 x n1

  # print(f"Diagonal blocks are \n {diag_blocks}")

  X11_mean = diag_blocks.mean(axis = 0)
  # print(f"Mean of diagonal blocks \n {X11_mean}")

  projected_C = C.copy()

  for i in range(n_blocks):
    projected_C[i*n1:(i+1)*n1,i*n1:(i+1)*n1] = X11_mean

  # print(f"Projected C is \n {projected_C}")

  return projected_C

def proj_positivecone(Y,epsilon):
  n = Y.shape[0]
  d, U = np.linalg.eigh(Y)
  dthreshold = np.maximum(d, epsilon)
  X = U@np.diag( dthreshold.flatten())@U.T
  X = (X+X.T)/2
  return X

def get_constant_block_cov_admm(C, epsilon = 1e-2, dimx = 672, n1 = 96, rho = 10, MAXITER = 1000, FREQ_PRINT = 1, eps_abs = 1e-3, eps_rel = 1e-8 ):

  """
  Input: a Covariance matrix C that has partitioned into Cxx,Cyy,Cxy,Cxy' blocks
  Assumption: Each Cxx and Cyy is partitioned into D blocks, each block has size n1 x n1
  Function: Project C onto the block Toeplitz and positive definite using the ADMM algorithm.
  Output: numpy.ndarray
      The estimated covariance matrix that has constant blocks diagonal in Cxx, Cyy, denoted as Z variable in ADMM

  Parameters
  ----------
  C : numpy.ndarray
      The sample covariance matrix (required input). Must be square and symmetric.
  epsilon: float
      A small value to adjust the minimum eigenvalue of C
  dimx: int
      The dimension of the main Cxx and Cyy blocks
  n1: int
      The dimension of each block in Cxx (number of blocks = dimx/n1 )
  rho : float, optional
      The augmented Lagrangian parameter. Controls the convergence speed and
      stability of the ADMM algorithm. Default is 10.
  MAXITER : int, optional
      The maximum number of ADMM iterations to run before stopping. Default is 1000.
  FREQ_PRINT : int, optional
      The frequency (in iterations) at which convergence status is printed
      to the console. Default is 1 (print every iteration).
  eps_abs : float, optional
      Absolute tolerance for the primal and dual residuals in the ADMM stop
      condition. Default is 1e-3.
  eps_rel : float, optional
      Relative tolerance for the primal and dual residuals. Default is 1e-8.

  """

  if ( not np.allclose(C, C.T, atol = 1e-3)) or (C.shape[0] != C.shape[1]):
    print("Error: Input C is NOT symmetric, OR C is NOT square")
    return

  n = C.shape[0]

  # Initialize variables
  X = C
  Z = C
  U = X-Z

  for k in range(MAXITER):

    # X-update (Project on Toeplitz)

    X = (1/(rho+1))*( C + rho*( Z - U ) )
    X11 = X[:dimx,:dimx]
    X22 = X[dimx:,dimx:]
    X[:dimx,:dimx] = project_constant_block(X11,n1)
    X[dimx:,dimx:] = project_constant_block(X22,n1)

    # Z-update (Project on positive cone)
    Zold = Z.copy()
    Z = proj_positivecone(X+U,epsilon)

    # U-update (residual sum)

    U = U+X-Z

    # Compute primal and dual residuals
    res_primal_norm = rho*np.linalg.norm(Z-X,'fro')
    res_dual_norm = rho*np.linalg.norm(Z-Zold,'fro')
    eps_dual = n*eps_abs + eps_rel*rho*np.linalg.norm(U,'fro')
    eps_primal = n*eps_abs + eps_rel*max(np.linalg.norm(X,'fro'), np.linalg.norm(Z,'fro'))

    if k % FREQ_PRINT == 0:
      print(f"Iteration:{k} Primal residual = {res_primal_norm:.2f}, Dual residual = {res_dual_norm:.2f} ")

    if (res_primal_norm <= eps_primal ) and ( res_dual_norm <= eps_dual):
      print(f" At iteration {k}, the algorithm succesfully converges to the desired tolerance")
      print(f"Mininum eigenvalues of X is {np.linalg.eigh(X)[0].min()}")
      print(f"Mininum eigenvalues of Z is {np.linalg.eigh(Z)[0].min()}")
      break

  # Return Z (instead of X) because Z is projected to positive cone (guarantee positiveness of covariance)
  # Z might not be exactly Toeplitz due to accuracy of ADMM

  return Z

# def get_constant_block_cov_cvxpy(C, epsilon = 1e-2, dimx = 672, n1 = 96):
def get_constant_block_cov_cvxpy(C, epsilon, dimx, n1, solver=None ):

  if dimx % n1 != 0 :
    print("Error: the dimension of C is not a multiple of n1, can't split the blocks")
  else:
    n_blocks = int(dimx//n1)

  # Solve the optimization problem
  n = int(dimx*2)
  S = cp.Variable( (n,n), symmetric = True)
  Sxx = cp.Variable( (dimx,dimx), symmetric = True)
  Syy = cp.Variable( (dimx,dimx), symmetric = True)
  diagonal_blocks_X = cp.Variable( (n1,n1), symmetric = True)
  diagonal_blocks_Y = cp.Variable( (n1,n1), symmetric = True)

  constraints = [S >> epsilon*np.eye(n)] # Add small Identity to make Sigma positive
  constraints.append( S[:dimx,:dimx] == Sxx )
  constraints.append( S[dimx:,dimx:] == Syy )
  for i in range(n_blocks):
    constraints.append( Sxx[i*n1:(i+1)*n1,i*n1:(i+1)*n1] == diagonal_blocks_X )
    constraints.append( Syy[i*n1:(i+1)*n1,i*n1:(i+1)*n1] == diagonal_blocks_Y )

  prob = cp.Problem( cp.Minimize ( cp.norm(C-S, "fro")), constraints)
  prob.solve()
  return S.value