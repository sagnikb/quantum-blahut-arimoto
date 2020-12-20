# quantum-blahut-arimoto
Python implementation of quantum Blahut-Arimoto algorithms to compute quantum information quantities (https://arxiv.org/abs/1905.01286)

::: {role="main"}
::: {#section-intro .section}
Expand source code

    import numpy as np
    import scipy.linalg as linalg
    import random
    import matplotlib.pyplot as plt

    def D(rho, sigma):
        '''
        Returns the quantum relative entropy between two density matrices rho and sigma.
        Does not check for ker(sigma) subseteq ker(rho) (in which case this value is inf) 
        '''
        return(np.trace(rho @ (linalg.logm(rho) - linalg.logm(sigma)))/(np.log(2)))

    def randpsd(n):
        '''
        Returns a random real psd matrix of dimension n x n, by first creating a random 
        square matrix M of dimension n and then returning M @ M^T, which is always psd
        after making the trace 1
        '''
        M = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                M[i,j] = random.random()
        M = M @ (M.T)
        return (1/(np.trace(M))) * M  

    def create_cq_channel(dim, n):
        '''
        Creates a random cq-channel with input alphabet size n and output dimension dim
        Uses randpsd
        '''
        channel = []
        for i in range(n):
            channel.append(randpsd(dim))
        return channel

    def create_basis(dim):
        '''
        Creates the standard basis for C^dim
        '''
        basis = []
        for i in range(dim):
            basis_vector = np.zeros((1, dim))
            basis_vector[0, i] = 1
            basis.append(basis_vector)
        return basis

    def create_amplitude_damping_channel(p):
        '''
        Returns Kraus operators for 2x2 amplitude damping channel with parameter p
        '''
        kraus_operators = []
        M = np.zeros((2,2)); M[0,0] = 1; M[1,1] = np.sqrt(1-p)
        kraus_operators.append(M)
        M = np.zeros((2,2)); M[0,1] = np.sqrt(p)
        kraus_operators.append(M)
        return(kraus_operators)

    def adjoint_channel(kraus_operators):
        '''
        Given a set of Kraus operators for a channel, returns the Kraus operators for 
        the adjoint channel
        '''
        adjoint_kraus_operators = []
        for matrix in kraus_operators:
            adjoint_kraus_operators.append(matrix.conj().T)
        return adjoint_kraus_operators

    def complementary_channel(kraus_operators):
        '''
        Given a set of Kraus operators for a channel, returns the Kraus operators for
        the complementary channel. First computes the Choi matrix for the Kraus operators,
        then computes eigenvalues and eigenvectors for the Choi matrix and then 'folds them'
        to create Kraus operators for the complementary channel
        (https://quantumcomputing.stackexchange.com/a/5797)
        '''
        n = len(kraus_operators)
        zbasis = create_basis(n)
        choi = np.zeros((np.square(n), np.square(n)))
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    for m in range(n):
                        choi = choi + np.trace(kraus_operators[m].conj().T @ kraus_operators[l] @ np.outer(zbasis[j], zbasis[k])) * \
                        np.kron(np.outer(zbasis[l], zbasis[m]), np.outer(zbasis[j], zbasis[k]))
        w, v = linalg.eigh(choi) #Choi matrix is symmetric, and eigh is more accurate
        v = v.T # the columns of V are the eigenvectors
        channel = []
        for i in range(len(w)):
            channel.append(np.sqrt(w[i])*np.resize(v[i], (n,n))) # folding to get the Kraus operators 
        return channel

    def act_channel(kraus_operators, density_matrix):
        '''
        Given a channel as a list of Kraus operators and an input density matrix,
        computes the output density matrix.
        '''
        l = len(kraus_operators)
        output_matrix = np.zeros(np.shape(density_matrix))
        for i in range(l):
            output_matrix = output_matrix + kraus_operators[i] @ density_matrix @ (kraus_operators[i].conj().T)
        return output_matrix

    def J(quantity, rho, sigma, gamma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
        '''
        Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be 'h', 
        'tc', 'coh' or 'qmi') taking as input the channel and the associated adj, complementary and adjoint
        complementary channels
        '''
        return -1*gamma*np.trace(rho @ (linalg.logm(rho)/np.log(2))) + np.trace(rho @ (gamma * (linalg.logm(sigma)/np.log(2)) + 
        F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)))

    def F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
        '''
        Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be 'h', 
        'tc', 'coh' or 'qmi') taking as input the channel and the associated adj, complementary and adjoint
        complementary channels
        '''
        if quantity == 'h':
            s =  np.shape(basis[0])
            output_matrix = np.zeros((s[1], s[1]))
            Esigma = np.zeros((np.shape(channel[0])[0], np.shape(channel[0])[0]))
            for i in range(len(channel)):
                Esigma = Esigma + sigma[i,i] * channel[i]
            for i in range(len(channel)):
                output_matrix = output_matrix + np.outer(basis[i], basis[i]) * np.trace(channel[i] @ (linalg.logm(channel[i])/np.log(2) - 
                linalg.logm(Esigma)/np.log(2)))
            return output_matrix
        elif quantity == 'tc':
            return -1*linalg.logm(sigma)/np.log(2) + act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
        elif quantity == 'coh':
            return act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
            act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
        elif quantity == 'qmi':
            return -1*linalg.logm(sigma)/np.log(2) + act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
            act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
        else:
            print('quantity not found')
            return 1

    def capacity(quantity, channel, gamma, dim, basis, eps, **kwargs):
        '''
        Runs the Blahut-Arimoto algorithm to compute the capacity given by 'quantity' (which can be 'h', 'tc', 
        'coh' or 'qmi' taking the channel, gamma, dim, basis and tolerance (eps) as inputs)
        With the optional keyword arguments 'plot' (Boolean), it outputs a plot showing how the calculated value 
        changes with the number of iterations.
        With the optional keyword arguments 'latexplot' (Boolean), the plot uses latex in the labels
        '''
        if quantity != 'h': #holevo quantity doesn't need the other channels
            Adjoint_channel = adjoint_channel(channel)
            Complementary_channel = complementary_channel(channel)
            Adj_Complementary_channel = adjoint_channel(complementary_channel(channel))
        else: 
            Adjoint_channel = channel; Complementary_channel = channel; Adj_Complementary_channel = channel
        #to store the calculated values
        itern = []
        value = []
        #initialization
        rhoa = np.diag((1/dim)*np.ones((1,dim))[0]) 
        #Blahut-Arimoto algorithm iteration
        for t in range(int(gamma*np.log2(dim)/eps)):
            itern.append(t)
            sigmab = rhoa 
            rhoa = linalg.expm(np.log(2)*(linalg.logm(sigmab)/np.log(2) + \
            (1/gamma)*F(quantity, sigmab, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel)))
            rhoa = rhoa/np.trace(rhoa)
            value.append(J(quantity, rhoa, rhoa, gamma, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel))
        #Plotting
        if kwargs['plot'] == True:
            if kwargs['latexplot'] == True:
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
            fig, ax = plt.subplots()
            plt.plot(itern, value, marker = '.', markersize='7', label = r'Capacity value vs iteration')
            plt.xlabel(r'Number of iterations', fontsize = '14')
            plt.ylabel(r'Value of capacity', fontsize = '14')
            plt.xticks(fontsize = '8')
            plt.yticks(fontsize = '8')
            plt.grid(True)
            plt.show()
        return J(quantity, rhoa, rhoa, gamma, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel)
:::

::: {.section}
:::

::: {.section}
:::

::: {.section}
Functions {#header-functions .section-title}
---------

` def D(rho, sigma)`{.name .flex}

:   ::: {.desc}
    Returns the quantum relative entropy between two density matrices
    rho and sigma. Does not check for ker(sigma) subseteq ker(rho) (in
    which case this value is inf)
    :::

    Expand source code

        def D(rho, sigma):
            '''
            Returns the quantum relative entropy between two density matrices rho and sigma.
            Does not check for ker(sigma) subseteq ker(rho) (in which case this value is inf) 
            '''
            return(np.trace(rho @ (linalg.logm(rho) - linalg.logm(sigma)))/(np.log(2)))

` def F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)`{.name .flex}

:   ::: {.desc}
    Computes the function J from <https://arxiv.org/abs/1905.01286> for
    the given quantity (which can be \'h\', \'tc\', \'coh\' or \'qmi\')
    taking as input the channel and the associated adj, complementary
    and adjoint complementary channels
    :::

    Expand source code

        def F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
            '''
            Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be 'h', 
            'tc', 'coh' or 'qmi') taking as input the channel and the associated adj, complementary and adjoint
            complementary channels
            '''
            if quantity == 'h':
                s =  np.shape(basis[0])
                output_matrix = np.zeros((s[1], s[1]))
                Esigma = np.zeros((np.shape(channel[0])[0], np.shape(channel[0])[0]))
                for i in range(len(channel)):
                    Esigma = Esigma + sigma[i,i] * channel[i]
                for i in range(len(channel)):
                    output_matrix = output_matrix + np.outer(basis[i], basis[i]) * np.trace(channel[i] @ (linalg.logm(channel[i])/np.log(2) - 
                    linalg.logm(Esigma)/np.log(2)))
                return output_matrix
            elif quantity == 'tc':
                return -1*linalg.logm(sigma)/np.log(2) + act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
            elif quantity == 'coh':
                return act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
                act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
            elif quantity == 'qmi':
                return -1*linalg.logm(sigma)/np.log(2) + act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
                act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
            else:
                print('quantity not found')
                return 1

` def J(quantity, rho, sigma, gamma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)`{.name .flex}

:   ::: {.desc}
    Computes the function J from <https://arxiv.org/abs/1905.01286> for
    the given quantity (which can be \'h\', \'tc\', \'coh\' or \'qmi\')
    taking as input the channel and the associated adj, complementary
    and adjoint complementary channels
    :::

    Expand source code

        def J(quantity, rho, sigma, gamma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
            '''
            Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be 'h', 
            'tc', 'coh' or 'qmi') taking as input the channel and the associated adj, complementary and adjoint
            complementary channels
            '''
            return -1*gamma*np.trace(rho @ (linalg.logm(rho)/np.log(2))) + np.trace(rho @ (gamma * (linalg.logm(sigma)/np.log(2)) + 
            F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)))

` def act_channel(kraus_operators, density_matrix)`{.name .flex}

:   ::: {.desc}
    Given a channel as a list of Kraus operators and an input density
    matrix, computes the output density matrix.
    :::

    Expand source code

        def act_channel(kraus_operators, density_matrix):
            '''
            Given a channel as a list of Kraus operators and an input density matrix,
            computes the output density matrix.
            '''
            l = len(kraus_operators)
            output_matrix = np.zeros(np.shape(density_matrix))
            for i in range(l):
                output_matrix = output_matrix + kraus_operators[i] @ density_matrix @ (kraus_operators[i].conj().T)
            return output_matrix

` def adjoint_channel(kraus_operators)`{.name .flex}

:   ::: {.desc}
    Given a set of Kraus operators for a channel, returns the Kraus
    operators for the adjoint channel
    :::

    Expand source code

        def adjoint_channel(kraus_operators):
            '''
            Given a set of Kraus operators for a channel, returns the Kraus operators for 
            the adjoint channel
            '''
            adjoint_kraus_operators = []
            for matrix in kraus_operators:
                adjoint_kraus_operators.append(matrix.conj().T)
            return adjoint_kraus_operators

` def capacity(quantity, channel, gamma, dim, basis, eps, **kwargs)`{.name .flex}

:   ::: {.desc}
    Runs the Blahut-Arimoto algorithm to compute the capacity given by
    \'quantity\' (which can be \'h\', \'tc\', \'coh\' or \'qmi\' taking
    the channel, gamma, dim, basis and tolerance (eps) as inputs) With
    the optional keyword arguments \'plot\' (Boolean), it outputs a plot
    showing how the calculated value changes with the number of
    iterations. With the optional keyword arguments \'latexplot\'
    (Boolean), the plot uses latex in the labels
    :::

    Expand source code

        def capacity(quantity, channel, gamma, dim, basis, eps, **kwargs):
            '''
            Runs the Blahut-Arimoto algorithm to compute the capacity given by 'quantity' (which can be 'h', 'tc', 
            'coh' or 'qmi' taking the channel, gamma, dim, basis and tolerance (eps) as inputs)
            With the optional keyword arguments 'plot' (Boolean), it outputs a plot showing how the calculated value 
            changes with the number of iterations.
            With the optional keyword arguments 'latexplot' (Boolean), the plot uses latex in the labels
            '''
            if quantity != 'h': #holevo quantity doesn't need the other channels
                Adjoint_channel = adjoint_channel(channel)
                Complementary_channel = complementary_channel(channel)
                Adj_Complementary_channel = adjoint_channel(complementary_channel(channel))
            else: 
                Adjoint_channel = channel; Complementary_channel = channel; Adj_Complementary_channel = channel
            #to store the calculated values
            itern = []
            value = []
            #initialization
            rhoa = np.diag((1/dim)*np.ones((1,dim))[0]) 
            #Blahut-Arimoto algorithm iteration
            for t in range(int(gamma*np.log2(dim)/eps)):
                itern.append(t)
                sigmab = rhoa 
                rhoa = linalg.expm(np.log(2)*(linalg.logm(sigmab)/np.log(2) + \
                (1/gamma)*F(quantity, sigmab, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel)))
                rhoa = rhoa/np.trace(rhoa)
                value.append(J(quantity, rhoa, rhoa, gamma, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel))
            #Plotting
            if kwargs['plot'] == True:
                if kwargs['latexplot'] == True:
                    plt.rc('text', usetex=True)
                    plt.rc('font', family='serif')
                fig, ax = plt.subplots()
                plt.plot(itern, value, marker = '.', markersize='7', label = r'Capacity value vs iteration')
                plt.xlabel(r'Number of iterations', fontsize = '14')
                plt.ylabel(r'Value of capacity', fontsize = '14')
                plt.xticks(fontsize = '8')
                plt.yticks(fontsize = '8')
                plt.grid(True)
                plt.show()
            return J(quantity, rhoa, rhoa, gamma, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel)

` def complementary_channel(kraus_operators)`{.name .flex}

:   ::: {.desc}
    Given a set of Kraus operators for a channel, returns the Kraus
    operators for the complementary channel. First computes the Choi
    matrix for the Kraus operators, then computes eigenvalues and
    eigenvectors for the Choi matrix and then \'folds them\' to create
    Kraus operators for the complementary channel
    (<https://quantumcomputing.stackexchange.com/a/5797>)
    :::

    Expand source code

        def complementary_channel(kraus_operators):
            '''
            Given a set of Kraus operators for a channel, returns the Kraus operators for
            the complementary channel. First computes the Choi matrix for the Kraus operators,
            then computes eigenvalues and eigenvectors for the Choi matrix and then 'folds them'
            to create Kraus operators for the complementary channel
            (https://quantumcomputing.stackexchange.com/a/5797)
            '''
            n = len(kraus_operators)
            zbasis = create_basis(n)
            choi = np.zeros((np.square(n), np.square(n)))
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        for m in range(n):
                            choi = choi + np.trace(kraus_operators[m].conj().T @ kraus_operators[l] @ np.outer(zbasis[j], zbasis[k])) * \
                            np.kron(np.outer(zbasis[l], zbasis[m]), np.outer(zbasis[j], zbasis[k]))
            w, v = linalg.eigh(choi) #Choi matrix is symmetric, and eigh is more accurate
            v = v.T # the columns of V are the eigenvectors
            channel = []
            for i in range(len(w)):
                channel.append(np.sqrt(w[i])*np.resize(v[i], (n,n))) # folding to get the Kraus operators 
            return channel

` def create_amplitude_damping_channel(p)`{.name .flex}

:   ::: {.desc}
    Returns Kraus operators for 2x2 amplitude damping channel with
    parameter p
    :::

    Expand source code

        def create_amplitude_damping_channel(p):
            '''
            Returns Kraus operators for 2x2 amplitude damping channel with parameter p
            '''
            kraus_operators = []
            M = np.zeros((2,2)); M[0,0] = 1; M[1,1] = np.sqrt(1-p)
            kraus_operators.append(M)
            M = np.zeros((2,2)); M[0,1] = np.sqrt(p)
            kraus_operators.append(M)
            return(kraus_operators)

` def create_basis(dim)`{.name .flex}

:   ::: {.desc}
    Creates the standard basis for C\^dim
    :::

    Expand source code

        def create_basis(dim):
            '''
            Creates the standard basis for C^dim
            '''
            basis = []
            for i in range(dim):
                basis_vector = np.zeros((1, dim))
                basis_vector[0, i] = 1
                basis.append(basis_vector)
            return basis

` def create_cq_channel(dim, n)`{.name .flex}

:   ::: {.desc}
    Creates a random cq-channel with input alphabet size n and output
    dimension dim Uses randpsd
    :::

    Expand source code

        def create_cq_channel(dim, n):
            '''
            Creates a random cq-channel with input alphabet size n and output dimension dim
            Uses randpsd
            '''
            channel = []
            for i in range(n):
                channel.append(randpsd(dim))
            return channel

` def randpsd(n)`{.name .flex}

:   ::: {.desc}
    Returns a random real psd matrix of dimension n x n, by first
    creating a random square matrix M of dimension n and then returning
    M @ M\^T, which is always psd after making the trace 1
    :::

    Expand source code

        def randpsd(n):
            '''
            Returns a random real psd matrix of dimension n x n, by first creating a random 
            square matrix M of dimension n and then returning M @ M^T, which is always psd
            after making the trace 1
            '''
            M = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    M[i,j] = random.random()
            M = M @ (M.T)
            return (1/(np.trace(M))) * M  
:::

::: {.section}
:::

Index
=====

::: {.toc}
:::

-   ### [Functions](#header-functions) {#functions}

    -   `D`
    -   `F`
    -   `J`
    -   `act_channel`
    -   `adjoint_channel`
    -   `capacity`
    -   `complementary_channel`
    -   `create_amplitude_damping_channel`
    -   `create_basis`
    -   `create_cq_channel`
    -   `randpsd`
:::

Generated by [pdoc 0.9.2](https://pdoc3.github.io/pdoc).
