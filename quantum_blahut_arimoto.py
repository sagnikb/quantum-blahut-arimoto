from __future__ import annotations
from typing import List
import random
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Kraus

# class DensityMatrix:

#     def __init__(self, mat: List[List[float]]) -> None:
#         self.mat = mat

#     def __add__(self, other: DensityMatrix) -> DensityMatrix:
#         return DensityMatrix(self.mat + other.mat)

#     def __sub__(self, other: DensityMatrix) -> DensityMatrix:
#         return DensityMatrix(self.mat - other.mat)

#     def D(self, rho: DensityMatrix) -> float:
#         """
#         Return the quantum relative entropy D(self || rho) between two density
#         matrices self and rho.
#         """
#         return np.trace(self.mat @ (linalg.logm(self.mat)
#                         - linalg.logm(rho.mat)))/(np.log(2))

#     @classmethod
#     def random_density_matrix(cls, dim: int) -> DensityMatrix:
#         """Return a random real psd matrix of dimension dim x dim"""
#         M = np.zeros((dim,dim))
#         for i in range(dim):
#             for j in range(dim):
#                 M[i,j] = random.random()
#         M = M @ (M.T)
#         return cls((1/(np.trace(M))) * M)

def random_density_matrix(dim: int) -> DensityMatrix:
    """Return a random real psd matrix of dimension dim x dim"""
    M = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            M[i,j] = random.random()
    M = M @ (M.T)
    return DensityMatrix((1/(np.trace(M))) * M)

class Channel:

    def __init__(self, kraus_operators: List[List[List[float]]]) -> None:
        self.kraus_operators = kraus_operators

    @staticmethod
    def create_basis(dim: int) -> List[List[float]]:
        """Creates the standard basis for C^dim"""
        basis = []
        for i in range(dim):
            basis_vector = np.zeros((1, dim))
            basis_vector[0, i] = 1.0
            basis.append(basis_vector)
        return basis

    @property
    def adjoint_channel(self) -> Channel:
        """Return the Kraus operators for the adjoint channel."""
        return Channel([matrix.conj().T for matrix in self.kraus_operators])

    @property
    def complementary_channel(self) -> Channel:
        """Returns the Kraus operators for the complementary channel.
        
        First computes the Choi matrix for the Kraus operators,then computes
        eigenvalues and eigenvectors for the Choi matrix and then 'folds them'
        to create Kraus operators for the complementary channel.
        (https://quantumcomputing.stackexchange.com/a/5797)
        """
        n = len(self.kraus_operators)
        zbasis = Channel.create_basis(n)
        choi = np.zeros((np.square(n), np.square(n)))
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    for m in range(n):
                        choi += (np.trace(
                            self.kraus_operators[m].conj().T
                            @ self.kraus_operators[l]
                            @ np.outer(zbasis[j], zbasis[k])
                            )
                            * np.kron(
                                np.outer(zbasis[l], zbasis[m]),
                                np.outer(zbasis[j], zbasis[k])
                            )
                            )
        # Choi matrix is symmetric, and eigh is more accurate
        w, v = linalg.eigh(choi)
        # The columns of V are the eigenvectors
        v = v.T
        channelops = []
        for i in range(len(w)):
            # folding to get the Kraus operators
            channelops.append(np.sqrt(w[i])*np.resize(v[i], (n,n)))
        return Channel(channelops)

    @property
    def adjoint_complementary_channel(self) -> Channel:
        """Return adjoint complementary channel"""
        comp_channel = self.complementary_channel
        return Channel([matrix.conj().T for matrix in comp_channel.kraus_operators])


class AmplitudeDampingChannel(Channel):

    def __init__(self, p: float, dim: int = 2) -> None:
        kraus_operators = []
        M = np.zeros((2,2))
        M[0,0] = 1
        M[1,1] = np.sqrt(1-p)
        kraus_operators.append(M)
        M = np.zeros((2,2))
        M[0,1] = np.sqrt(p)
        kraus_operators.append(M)
        super().__init__(kraus_operators)


class CqChannel(Channel):

    def __init__(self,
                density_matrices: List[DensityMatrix],
            ) -> None:
        # Check definition of Kraus operators for cq-channel
        super().__init__(density_matrices)

    @classmethod
    def random_cq_channel(cls, ip_alph: int, op_dim: int):
        density_matrices = []
        for _ in range(ip_alph):
            density_matrices.append(
                random_density_matrix(op_dim).data
            )
        return(cls(density_matrices))


def act_channel(
    channel: Channel,
    density_matrix: DensityMatrix
    ) -> DensityMatrix:
    """
    Given a channel as a list of Kraus operators and an input density matrix,
    computes the output density matrix.
    """
    output_matrix = np.zeros(np.shape(density_matrix.data))
    for op in channel.kraus_operators:
        output_matrix += (op @ density_matrix.data @ (op.conj().T))
    return DensityMatrix(output_matrix)

def J(
    quantity: str,
    rho: DensityMatrix,
    sigma: DensityMatrix,
    gamma: float,
    basis: List[List[float]],
    channel: Channel) -> float:
    """
    Computes the function J from https://arxiv.org/abs/1905.01286 for the given
    quantity (which can be 'h', 'tc', 'coh' or 'qmi') taking as input the
    channel.
    """
    return (-1*gamma*np.trace(rho.data @ (linalg.logm(rho.data)/np.log(2)))
        + np.trace(
            rho.data @ (gamma * (linalg.logm(sigma.data)/np.log(2))
            + F(quantity, sigma, basis, channel).data)
            )
        )

def F(
    quantity: str,
    sigma: DensityMatrix,
    basis: List[List[float]],
    channel: Channel) -> DensityMatrix:
    """
    Computes the function J from https://arxiv.org/abs/1905.01286 for the given
    quantity (which can be 'h', 'tc', 'coh' or 'qmi') taking as input the
    channel.
    """

    if quantity == 'h':
        size =  np.shape(basis[0])
        output_matrix = np.zeros((size[1], size[1]))
        e_sigma = np.zeros((np.shape(channel.kraus_operators[0])[0],
                        np.shape(channel.kraus_operators[0])[0])
                        )
        for i in range(len(channel.kraus_operators)):
            e_sigma += sigma.data[i,i] * channel.kraus_operators[i]
        for i in range(len(channel.kraus_operators)):
            output_matrix += (np.outer(basis[i], basis[i])
                            * np.trace(channel.kraus_operators[i]
                            @ (linalg.logm(channel.kraus_operators[i])/np.log(2)
                                - linalg.logm(e_sigma)/np.log(2)
                                )
                            )
                        )
        return DensityMatrix(output_matrix)

    if quantity == 'tc':
        temp = act_channel(
                        channel.adjoint_channel,
                        DensityMatrix(linalg.logm(
                            act_channel(channel, sigma).data)/np.log(2)
                        )
                    )
        return DensityMatrix(-1*(linalg.logm(sigma.data)/np.log(2))) + temp
        
    if quantity == 'coh':
        temp1 = act_channel(
                        channel.adjoint_complementary_channel,
                        DensityMatrix(linalg.logm(
                            act_channel(channel.complementary_channel, sigma).data
                        )/np.log(2))
            )
        temp2 = act_channel(
                        channel.adjoint_channel,
                        DensityMatrix(linalg.logm(
                            act_channel(channel, sigma).data
                        )/np.log(2))
            )
        return temp1 - temp2

    if quantity == 'qmi':
        temp1 = act_channel(
                        channel.adjoint_complementary_channel,
                        DensityMatrix(linalg.logm(
                            act_channel(channel.complementary_channel, sigma).data
                        )/np.log(2))
            )
        temp2 = act_channel(
                        channel.adjoint_channel,
                        DensityMatrix(linalg.logm(
                            act_channel(channel, sigma).data
                        )/np.log(2))
            )
        return DensityMatrix(-1*linalg.logm(sigma.data)/np.log(2)) + temp1 - temp2

    print('quantity not found')
    return 1

def capacity(quantity, channel, gamma, dim, basis, eps, **kwargs):
    """
    Runs the Blahut-Arimoto algorithm to compute the capacity given by
    'quantity' (which can be 'h', 'tc', 'coh' or 'qmi' taking the channel,
    gamma, dim, basis and tolerance (eps) as inputs).
    With the optional keyword arguments 'plot' (Boolean), it outputs a plot
    showing how the calculated value changes with the number of iterations.
    With the optional keyword arguments 'latexplot' (Boolean), the plot uses
    latex in the labels
    """
    #to store the calculated values
    itern = []
    value = []
    #initialization
    rhoa = DensityMatrix(np.diag((1/dim)*np.ones((1,dim))[0]))
    #Blahut-Arimoto algorithm iteration
    for iterator in range(int(gamma*np.log2(dim)/eps)):
    # for iterator in range(1):
        itern.append(iterator)
        sigmab = rhoa
        rhoa = linalg.expm(np.log(2)*(linalg.logm(sigmab.data)/np.log(2)
                        + (1/gamma)*(F(quantity, sigmab, basis, channel).data)))
        rhoa = DensityMatrix(rhoa/np.trace(rhoa))
        value.append(J(quantity, rhoa, rhoa, gamma, basis, channel))
    #Plotting
    if kwargs['plot'] is True:
        # if kwargs['latexplot'] is True:
        #     plt.rc('text', usetex=True)
        #     plt.rc('font', family='serif')
        fig, ax = plt.subplots()
        plt.plot(itern, value,
            marker = '.',
            markersize='7',
            label = r'Capacity value vs iteration'
        )
        plt.xlabel(r'Number of iterations', fontsize = '14')
        plt.ylabel(r'Value of capacity', fontsize = '14')
        plt.xticks(fontsize = '8')
        plt.yticks(fontsize = '8')
        plt.grid(True)
        plt.show()
    return J(quantity, rhoa, rhoa, gamma, basis, channel)

def main():
    adc = AmplitudeDampingChannel(0.3)
    # cqc = CqChannel.random_cq_channel(2,2)
    print(capacity('tc', adc, 0.8, 2, Channel.create_basis(2), 0.001, plot = True))
    # print(capacity('h', cqc, 0.8, 2, Channel.create_basis(2), 0.001, plot = True))

if __name__ == "__main__":
    main()
