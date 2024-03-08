import numpy as np

def linear_solver(A, b):
    """
    Solve for x Ax=b. Assume A is invertible.
    Args:
        A: nxn numpy array
        b: 0xn numpy array

    Returns:
        x: 0xn numpy array
    """

    # Function to get the inverse of a matrix
    def inverse(A):
        # Function to create a lower order matrix by deleting ith row and jth column
        def _low_order_matrix(A, row, column):
            # delete the ith row
            A_low_order = np.delete(A, row, axis=0)
            # delete the jth row
            A_low_order = np.delete(A_low_order, column, axis=1)
            return A_low_order
        
        # Function to find the determinant of A
        def _determinant(A):
            ################ RECUSION METHOD ########################
            # NOTE: This method is having issues with the autograder for the case
            # linear_solver_00. Possible reason could be the recusion method is
            # causing the floating point errors to multiply as recusion progresses.
            # #######################################################
            #  
            # if A.shape[0] == 1 and A.shape[1] == 1:
            #     return A[0][0]
            # else:
            #     # find the determinant by expanding along the first row of A
            #     i = 0 # index for the first row
            #     det = 0.0 # initialize the variable 'det'

            #     for j in range(A.shape[1]):
            #         det +=(-1)**(i+j)*A[i][j]*_determinant(_low_order_matrix(A,i,j))
            #     return det

            ############### NUMPY BUILT-IN FUNC. ####################
            return np.linalg.det(A)
            
        # Function to find the adjoint of A
        def _adj(A):
            # define a cofactor matrix with zeros
            c_A = np.zeros(shape=(A.shape[0], A.shape[1]))
            
            # update the cofactor matrix with actual values
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    c_A[i,j] = (-1)**(i+j)*_determinant(_low_order_matrix(A,i,j))
            
            # get adjoint of A by transposing cofactor matrix
            adj_A = np.transpose(c_A)
            return adj_A

        # Calculate the inverse of A using above helper functions
        # Use Moore-Penrose inverse method: A^+ = (A^T * A)^-1 * A^T
        inverse_A = (1/_determinant(np.matmul(np.transpose(A),A)))* \
                    _adj(np.matmul(np.transpose(A),A))
        inverse_A = np.matmul(inverse_A, np.transpose(A))

        return inverse_A
        
    x = np.matmul(inverse(A), b)
    return x

def angle_solver(v1, v2):
    """
    Solves for the magnitude of the angle between v1 and v2
    Args:
        v1: 0xn numpy array
        v2: 0xn numpy array

    Returns:
        theta = scalar >= 0 = angle in radians
    """

    # Get dimensions of the vectors (size is used since vectors are 1D arrays)
    v1_size = v1.size
    v2_size = v2.size
    
    # Calcalate the dot product and the magnitude of the vectors
    if v1_size == v2_size:
        # dot product (np.dot can be used as alternative)
        dot_v1v2 = np.sum([v1[i]*v2[j] for i,j in zip(range(v1_size), range(v2_size))])
        
        # magnitudes
        mag_v1 = np.sqrt(np.sum(np.power(v1,2)))
        mag_v2 = np.sqrt(np.sum(np.power(v2,2)))
        
        # angle between v1 and v2
        theta = np.arccos(dot_v1v2/(mag_v1*mag_v2))
        return theta
    
    # If the two vectors are of different sizes, raise an error
    else:
        raise Exception("The two vectors are of different sizes")

def linear_euler_integration(A, x0, dt, nSteps):
    """
    Integrate the ode x'=Ax using euler integration where:
    x_{k+1} = dt (A x_k) + x_k
    Args:
        A: nxn np array describing linear differential equation
        x0: 0xn np array Initial condition
        dt: scalar, time step
        nSteps: scalar, number of time steps

    Returns:
        x: state after nSteps time steps (np array)
    """

    # Use a for loop to start from x0 and calculate x_{k+1} after nSteps
    for k in range(nSteps):
        if k==0:
            # initialize x_k for initial condition
            x_k = x0
            # use 'reshape' to make x_k compatible for matrix multiplication
            x_k = x_k.reshape(x_k.size, 1)
        
        if (x_k.shape[0] == A.shape[1]):
            # update x_{k+1}
            x_k_plus_1 = x_k + dt*np.matmul(A, x_k)
            # update x_k for next iteration
            x_k = x_k_plus_1
        else:
            raise Exception("A & x0 are not compatible for matrix multiplication")
    
    # Remove axis of length 1 from x_{k+1}
    return np.squeeze(x_k_plus_1)


if __name__ == '__main__':
    # Example call for linear solver
    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    print(linear_solver(A, b))

    # Example call for angles between vectors
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    print(angle_solver(v1, v2))

    # Example call for euler integration
    A = np.random.rand(3, 3)
    x0 = np.array([1, 1, 1])
    dt = 0.01
    nSteps = 100
    print(linear_euler_integration(A, x0, dt, nSteps))
