<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
<meta name="generator" content="pdoc 0.9.2" />
<title>quantum_blahut_arimoto API documentation</title>
<meta name="description" content="" />
<link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/sanitize.min.css" integrity="sha256-PK9q560IAAa6WVRRh76LtCaI8pjTJ2z11v0miyNNjrs=" crossorigin>
<link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/typography.min.css" integrity="sha256-7l/o7C8jubJiy74VsKTidCy1yBkRtiUGbVkYBylBqUg=" crossorigin>
<link rel="stylesheet preload" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/styles/github.min.css" crossorigin>
<style>:root{--highlight-color:#fe9}.flex{display:flex !important}body{line-height:1.5em}#content{padding:20px}#sidebar{padding:30px;overflow:hidden}#sidebar > *:last-child{margin-bottom:2cm}.http-server-breadcrumbs{font-size:130%;margin:0 0 15px 0}#footer{font-size:.75em;padding:5px 30px;border-top:1px solid #ddd;text-align:right}#footer p{margin:0 0 0 1em;display:inline-block}#footer p:last-child{margin-right:30px}h1,h2,h3,h4,h5{font-weight:300}h1{font-size:2.5em;line-height:1.1em}h2{font-size:1.75em;margin:1em 0 .50em 0}h3{font-size:1.4em;margin:25px 0 10px 0}h4{margin:0;font-size:105%}h1:target,h2:target,h3:target,h4:target,h5:target,h6:target{background:var(--highlight-color);padding:.2em 0}a{color:#058;text-decoration:none;transition:color .3s ease-in-out}a:hover{color:#e82}.title code{font-weight:bold}h2[id^="header-"]{margin-top:2em}.ident{color:#900}pre code{background:#f8f8f8;font-size:.8em;line-height:1.4em}code{background:#f2f2f1;padding:1px 4px;overflow-wrap:break-word}h1 code{background:transparent}pre{background:#f8f8f8;border:0;border-top:1px solid #ccc;border-bottom:1px solid #ccc;margin:1em 0;padding:1ex}#http-server-module-list{display:flex;flex-flow:column}#http-server-module-list div{display:flex}#http-server-module-list dt{min-width:10%}#http-server-module-list p{margin-top:0}.toc ul,#index{list-style-type:none;margin:0;padding:0}#index code{background:transparent}#index h3{border-bottom:1px solid #ddd}#index ul{padding:0}#index h4{margin-top:.6em;font-weight:bold}@media (min-width:200ex){#index .two-column{column-count:2}}@media (min-width:300ex){#index .two-column{column-count:3}}dl{margin-bottom:2em}dl dl:last-child{margin-bottom:4em}dd{margin:0 0 1em 3em}#header-classes + dl > dd{margin-bottom:3em}dd dd{margin-left:2em}dd p{margin:10px 0}.name{background:#eee;font-weight:bold;font-size:.85em;padding:5px 10px;display:inline-block;min-width:40%}.name:hover{background:#e0e0e0}dt:target .name{background:var(--highlight-color)}.name > span:first-child{white-space:nowrap}.name.class > span:nth-child(2){margin-left:.4em}.inherited{color:#999;border-left:5px solid #eee;padding-left:1em}.inheritance em{font-style:normal;font-weight:bold}.desc h2{font-weight:400;font-size:1.25em}.desc h3{font-size:1em}.desc dt code{background:inherit}.source summary,.git-link-div{color:#666;text-align:right;font-weight:400;font-size:.8em;text-transform:uppercase}.source summary > *{white-space:nowrap;cursor:pointer}.git-link{color:inherit;margin-left:1em}.source pre{max-height:500px;overflow:auto;margin:0}.source pre code{font-size:12px;overflow:visible}.hlist{list-style:none}.hlist li{display:inline}.hlist li:after{content:',\2002'}.hlist li:last-child:after{content:none}.hlist .hlist{display:inline;padding-left:1em}img{max-width:100%}td{padding:0 .5em}.admonition{padding:.1em .5em;margin-bottom:1em}.admonition-title{font-weight:bold}.admonition.note,.admonition.info,.admonition.important{background:#aef}.admonition.todo,.admonition.versionadded,.admonition.tip,.admonition.hint{background:#dfd}.admonition.warning,.admonition.versionchanged,.admonition.deprecated{background:#fd4}.admonition.error,.admonition.danger,.admonition.caution{background:lightpink}</style>
<style media="screen and (min-width: 700px)">@media screen and (min-width:700px){#sidebar{width:30%;height:100vh;overflow:auto;position:sticky;top:0}#content{width:70%;max-width:100ch;padding:3em 4em;border-left:1px solid #ddd}pre code{font-size:1em}.item .name{font-size:1em}main{display:flex;flex-direction:row-reverse;justify-content:flex-end}.toc ul ul,#index ul{padding-left:1.5em}.toc > ul > li{margin-top:.5em}}</style>
<style media="print">@media print{#sidebar h1{page-break-before:always}.source{display:none}}@media print{*{background:transparent !important;color:#000 !important;box-shadow:none !important;text-shadow:none !important}a[href]:after{content:" (" attr(href) ")";font-size:90%}a[href][title]:after{content:none}abbr[title]:after{content:" (" attr(title) ")"}.ir a:after,a[href^="javascript:"]:after,a[href^="#"]:after{content:""}pre,blockquote{border:1px solid #999;page-break-inside:avoid}thead{display:table-header-group}tr,img{page-break-inside:avoid}img{max-width:100% !important}@page{margin:0.5cm}p,h2,h3{orphans:3;widows:3}h1,h2,h3,h4,h5,h6{page-break-after:avoid}}</style>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/highlight.min.js" integrity="sha256-Uv3H6lx7dJmRfRvH8TH6kJD1TSK1aFcwgx+mdg3epi8=" crossorigin></script>
<script>window.addEventListener('DOMContentLoaded', () => hljs.initHighlighting())</script>
</head>
<body>
<main>
<article id="content">
<header>
<h1 class="title">Module <code>quantum_blahut_arimoto</code></h1>
</header>
<section id="section-intro">
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">import numpy as np
import scipy.linalg as linalg
import random
import matplotlib.pyplot as plt

def D(rho, sigma):
    &#39;&#39;&#39;
    Returns the quantum relative entropy between two density matrices rho and sigma.
    Does not check for ker(sigma) subseteq ker(rho) (in which case this value is inf) 
    &#39;&#39;&#39;
    return(np.trace(rho @ (linalg.logm(rho) - linalg.logm(sigma)))/(np.log(2)))

def randpsd(n):
    &#39;&#39;&#39;
    Returns a random real psd matrix of dimension n x n, by first creating a random 
    square matrix M of dimension n and then returning M @ M^T, which is always psd
    after making the trace 1
    &#39;&#39;&#39;
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j] = random.random()
    M = M @ (M.T)
    return (1/(np.trace(M))) * M  

def create_cq_channel(dim, n):
    &#39;&#39;&#39;
    Creates a random cq-channel with input alphabet size n and output dimension dim
    Uses randpsd
    &#39;&#39;&#39;
    channel = []
    for i in range(n):
        channel.append(randpsd(dim))
    return channel

def create_basis(dim):
    &#39;&#39;&#39;
    Creates the standard basis for C^dim
    &#39;&#39;&#39;
    basis = []
    for i in range(dim):
        basis_vector = np.zeros((1, dim))
        basis_vector[0, i] = 1
        basis.append(basis_vector)
    return basis

def create_amplitude_damping_channel(p):
    &#39;&#39;&#39;
    Returns Kraus operators for 2x2 amplitude damping channel with parameter p
    &#39;&#39;&#39;
    kraus_operators = []
    M = np.zeros((2,2)); M[0,0] = 1; M[1,1] = np.sqrt(1-p)
    kraus_operators.append(M)
    M = np.zeros((2,2)); M[0,1] = np.sqrt(p)
    kraus_operators.append(M)
    return(kraus_operators)

def adjoint_channel(kraus_operators):
    &#39;&#39;&#39;
    Given a set of Kraus operators for a channel, returns the Kraus operators for 
    the adjoint channel
    &#39;&#39;&#39;
    adjoint_kraus_operators = []
    for matrix in kraus_operators:
        adjoint_kraus_operators.append(matrix.conj().T)
    return adjoint_kraus_operators

def complementary_channel(kraus_operators):
    &#39;&#39;&#39;
    Given a set of Kraus operators for a channel, returns the Kraus operators for
    the complementary channel. First computes the Choi matrix for the Kraus operators,
    then computes eigenvalues and eigenvectors for the Choi matrix and then &#39;folds them&#39;
    to create Kraus operators for the complementary channel
    (https://quantumcomputing.stackexchange.com/a/5797)
    &#39;&#39;&#39;
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
    &#39;&#39;&#39;
    Given a channel as a list of Kraus operators and an input density matrix,
    computes the output density matrix.
    &#39;&#39;&#39;
    l = len(kraus_operators)
    output_matrix = np.zeros(np.shape(density_matrix))
    for i in range(l):
        output_matrix = output_matrix + kraus_operators[i] @ density_matrix @ (kraus_operators[i].conj().T)
    return output_matrix

def J(quantity, rho, sigma, gamma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
    &#39;&#39;&#39;
    Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be &#39;h&#39;, 
    &#39;tc&#39;, &#39;coh&#39; or &#39;qmi&#39;) taking as input the channel and the associated adj, complementary and adjoint
    complementary channels
    &#39;&#39;&#39;
    return -1*gamma*np.trace(rho @ (linalg.logm(rho)/np.log(2))) + np.trace(rho @ (gamma * (linalg.logm(sigma)/np.log(2)) + 
    F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)))

def F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
    &#39;&#39;&#39;
    Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be &#39;h&#39;, 
    &#39;tc&#39;, &#39;coh&#39; or &#39;qmi&#39;) taking as input the channel and the associated adj, complementary and adjoint
    complementary channels
    &#39;&#39;&#39;
    if quantity == &#39;h&#39;:
        s =  np.shape(basis[0])
        output_matrix = np.zeros((s[1], s[1]))
        Esigma = np.zeros((np.shape(channel[0])[0], np.shape(channel[0])[0]))
        for i in range(len(channel)):
            Esigma = Esigma + sigma[i,i] * channel[i]
        for i in range(len(channel)):
            output_matrix = output_matrix + np.outer(basis[i], basis[i]) * np.trace(channel[i] @ (linalg.logm(channel[i])/np.log(2) - 
            linalg.logm(Esigma)/np.log(2)))
        return output_matrix
    elif quantity == &#39;tc&#39;:
        return -1*linalg.logm(sigma)/np.log(2) + act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
    elif quantity == &#39;coh&#39;:
        return act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
        act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
    elif quantity == &#39;qmi&#39;:
        return -1*linalg.logm(sigma)/np.log(2) + act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
        act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
    else:
        print(&#39;quantity not found&#39;)
        return 1

def capacity(quantity, channel, gamma, dim, basis, eps, **kwargs):
    &#39;&#39;&#39;
    Runs the Blahut-Arimoto algorithm to compute the capacity given by &#39;quantity&#39; (which can be &#39;h&#39;, &#39;tc&#39;, 
    &#39;coh&#39; or &#39;qmi&#39; taking the channel, gamma, dim, basis and tolerance (eps) as inputs)
    With the optional keyword arguments &#39;plot&#39; (Boolean), it outputs a plot showing how the calculated value 
    changes with the number of iterations.
    With the optional keyword arguments &#39;latexplot&#39; (Boolean), the plot uses latex in the labels
    &#39;&#39;&#39;
    if quantity != &#39;h&#39;: #holevo quantity doesn&#39;t need the other channels
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
    if kwargs[&#39;plot&#39;] == True:
        if kwargs[&#39;latexplot&#39;] == True:
            plt.rc(&#39;text&#39;, usetex=True)
            plt.rc(&#39;font&#39;, family=&#39;serif&#39;)
        fig, ax = plt.subplots()
        plt.plot(itern, value, marker = &#39;.&#39;, markersize=&#39;7&#39;, label = r&#39;Capacity value vs iteration&#39;)
        plt.xlabel(r&#39;Number of iterations&#39;, fontsize = &#39;14&#39;)
        plt.ylabel(r&#39;Value of capacity&#39;, fontsize = &#39;14&#39;)
        plt.xticks(fontsize = &#39;8&#39;)
        plt.yticks(fontsize = &#39;8&#39;)
        plt.grid(True)
        plt.show()
    return J(quantity, rhoa, rhoa, gamma, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel)</code></pre>
</details>
</section>
<section>
</section>
<section>
</section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="quantum_blahut_arimoto.D"><code class="name flex">
<span>def <span class="ident">D</span></span>(<span>rho, sigma)</span>
</code></dt>
<dd>
<div class="desc"><p>Returns the quantum relative entropy between two density matrices rho and sigma.
Does not check for ker(sigma) subseteq ker(rho) (in which case this value is inf)</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def D(rho, sigma):
    &#39;&#39;&#39;
    Returns the quantum relative entropy between two density matrices rho and sigma.
    Does not check for ker(sigma) subseteq ker(rho) (in which case this value is inf) 
    &#39;&#39;&#39;
    return(np.trace(rho @ (linalg.logm(rho) - linalg.logm(sigma)))/(np.log(2)))</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.F"><code class="name flex">
<span>def <span class="ident">F</span></span>(<span>quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)</span>
</code></dt>
<dd>
<div class="desc"><p>Computes the function J from <a href="https://arxiv.org/abs/1905.01286">https://arxiv.org/abs/1905.01286</a> for the given quantity (which can be 'h',
'tc', 'coh' or 'qmi') taking as input the channel and the associated adj, complementary and adjoint
complementary channels</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
    &#39;&#39;&#39;
    Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be &#39;h&#39;, 
    &#39;tc&#39;, &#39;coh&#39; or &#39;qmi&#39;) taking as input the channel and the associated adj, complementary and adjoint
    complementary channels
    &#39;&#39;&#39;
    if quantity == &#39;h&#39;:
        s =  np.shape(basis[0])
        output_matrix = np.zeros((s[1], s[1]))
        Esigma = np.zeros((np.shape(channel[0])[0], np.shape(channel[0])[0]))
        for i in range(len(channel)):
            Esigma = Esigma + sigma[i,i] * channel[i]
        for i in range(len(channel)):
            output_matrix = output_matrix + np.outer(basis[i], basis[i]) * np.trace(channel[i] @ (linalg.logm(channel[i])/np.log(2) - 
            linalg.logm(Esigma)/np.log(2)))
        return output_matrix
    elif quantity == &#39;tc&#39;:
        return -1*linalg.logm(sigma)/np.log(2) + act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
    elif quantity == &#39;coh&#39;:
        return act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
        act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
    elif quantity == &#39;qmi&#39;:
        return -1*linalg.logm(sigma)/np.log(2) + act_channel(adj_complementary_channel, linalg.logm(act_channel(complementary_channel, sigma))/np.log(2)) - \
        act_channel(adjoint_channel, linalg.logm(act_channel(channel, sigma))/np.log(2))
    else:
        print(&#39;quantity not found&#39;)
        return 1</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.J"><code class="name flex">
<span>def <span class="ident">J</span></span>(<span>quantity, rho, sigma, gamma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)</span>
</code></dt>
<dd>
<div class="desc"><p>Computes the function J from <a href="https://arxiv.org/abs/1905.01286">https://arxiv.org/abs/1905.01286</a> for the given quantity (which can be 'h',
'tc', 'coh' or 'qmi') taking as input the channel and the associated adj, complementary and adjoint
complementary channels</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def J(quantity, rho, sigma, gamma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel):
    &#39;&#39;&#39;
    Computes the function J from https://arxiv.org/abs/1905.01286 for the given quantity (which can be &#39;h&#39;, 
    &#39;tc&#39;, &#39;coh&#39; or &#39;qmi&#39;) taking as input the channel and the associated adj, complementary and adjoint
    complementary channels
    &#39;&#39;&#39;
    return -1*gamma*np.trace(rho @ (linalg.logm(rho)/np.log(2))) + np.trace(rho @ (gamma * (linalg.logm(sigma)/np.log(2)) + 
    F(quantity, sigma, basis, channel, adjoint_channel, complementary_channel, adj_complementary_channel)))</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.act_channel"><code class="name flex">
<span>def <span class="ident">act_channel</span></span>(<span>kraus_operators, density_matrix)</span>
</code></dt>
<dd>
<div class="desc"><p>Given a channel as a list of Kraus operators and an input density matrix,
computes the output density matrix.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def act_channel(kraus_operators, density_matrix):
    &#39;&#39;&#39;
    Given a channel as a list of Kraus operators and an input density matrix,
    computes the output density matrix.
    &#39;&#39;&#39;
    l = len(kraus_operators)
    output_matrix = np.zeros(np.shape(density_matrix))
    for i in range(l):
        output_matrix = output_matrix + kraus_operators[i] @ density_matrix @ (kraus_operators[i].conj().T)
    return output_matrix</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.adjoint_channel"><code class="name flex">
<span>def <span class="ident">adjoint_channel</span></span>(<span>kraus_operators)</span>
</code></dt>
<dd>
<div class="desc"><p>Given a set of Kraus operators for a channel, returns the Kraus operators for
the adjoint channel</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def adjoint_channel(kraus_operators):
    &#39;&#39;&#39;
    Given a set of Kraus operators for a channel, returns the Kraus operators for 
    the adjoint channel
    &#39;&#39;&#39;
    adjoint_kraus_operators = []
    for matrix in kraus_operators:
        adjoint_kraus_operators.append(matrix.conj().T)
    return adjoint_kraus_operators</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.capacity"><code class="name flex">
<span>def <span class="ident">capacity</span></span>(<span>quantity, channel, gamma, dim, basis, eps, **kwargs)</span>
</code></dt>
<dd>
<div class="desc"><p>Runs the Blahut-Arimoto algorithm to compute the capacity given by 'quantity' (which can be 'h', 'tc',
'coh' or 'qmi' taking the channel, gamma, dim, basis and tolerance (eps) as inputs)
With the optional keyword arguments 'plot' (Boolean), it outputs a plot showing how the calculated value
changes with the number of iterations.
With the optional keyword arguments 'latexplot' (Boolean), the plot uses latex in the labels</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def capacity(quantity, channel, gamma, dim, basis, eps, **kwargs):
    &#39;&#39;&#39;
    Runs the Blahut-Arimoto algorithm to compute the capacity given by &#39;quantity&#39; (which can be &#39;h&#39;, &#39;tc&#39;, 
    &#39;coh&#39; or &#39;qmi&#39; taking the channel, gamma, dim, basis and tolerance (eps) as inputs)
    With the optional keyword arguments &#39;plot&#39; (Boolean), it outputs a plot showing how the calculated value 
    changes with the number of iterations.
    With the optional keyword arguments &#39;latexplot&#39; (Boolean), the plot uses latex in the labels
    &#39;&#39;&#39;
    if quantity != &#39;h&#39;: #holevo quantity doesn&#39;t need the other channels
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
    if kwargs[&#39;plot&#39;] == True:
        if kwargs[&#39;latexplot&#39;] == True:
            plt.rc(&#39;text&#39;, usetex=True)
            plt.rc(&#39;font&#39;, family=&#39;serif&#39;)
        fig, ax = plt.subplots()
        plt.plot(itern, value, marker = &#39;.&#39;, markersize=&#39;7&#39;, label = r&#39;Capacity value vs iteration&#39;)
        plt.xlabel(r&#39;Number of iterations&#39;, fontsize = &#39;14&#39;)
        plt.ylabel(r&#39;Value of capacity&#39;, fontsize = &#39;14&#39;)
        plt.xticks(fontsize = &#39;8&#39;)
        plt.yticks(fontsize = &#39;8&#39;)
        plt.grid(True)
        plt.show()
    return J(quantity, rhoa, rhoa, gamma, basis, channel, Adjoint_channel, Complementary_channel, Adj_Complementary_channel)</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.complementary_channel"><code class="name flex">
<span>def <span class="ident">complementary_channel</span></span>(<span>kraus_operators)</span>
</code></dt>
<dd>
<div class="desc"><p>Given a set of Kraus operators for a channel, returns the Kraus operators for
the complementary channel. First computes the Choi matrix for the Kraus operators,
then computes eigenvalues and eigenvectors for the Choi matrix and then 'folds them'
to create Kraus operators for the complementary channel
(<a href="https://quantumcomputing.stackexchange.com/a/5797">https://quantumcomputing.stackexchange.com/a/5797</a>)</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def complementary_channel(kraus_operators):
    &#39;&#39;&#39;
    Given a set of Kraus operators for a channel, returns the Kraus operators for
    the complementary channel. First computes the Choi matrix for the Kraus operators,
    then computes eigenvalues and eigenvectors for the Choi matrix and then &#39;folds them&#39;
    to create Kraus operators for the complementary channel
    (https://quantumcomputing.stackexchange.com/a/5797)
    &#39;&#39;&#39;
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
    return channel</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.create_amplitude_damping_channel"><code class="name flex">
<span>def <span class="ident">create_amplitude_damping_channel</span></span>(<span>p)</span>
</code></dt>
<dd>
<div class="desc"><p>Returns Kraus operators for 2x2 amplitude damping channel with parameter p</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def create_amplitude_damping_channel(p):
    &#39;&#39;&#39;
    Returns Kraus operators for 2x2 amplitude damping channel with parameter p
    &#39;&#39;&#39;
    kraus_operators = []
    M = np.zeros((2,2)); M[0,0] = 1; M[1,1] = np.sqrt(1-p)
    kraus_operators.append(M)
    M = np.zeros((2,2)); M[0,1] = np.sqrt(p)
    kraus_operators.append(M)
    return(kraus_operators)</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.create_basis"><code class="name flex">
<span>def <span class="ident">create_basis</span></span>(<span>dim)</span>
</code></dt>
<dd>
<div class="desc"><p>Creates the standard basis for C^dim</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def create_basis(dim):
    &#39;&#39;&#39;
    Creates the standard basis for C^dim
    &#39;&#39;&#39;
    basis = []
    for i in range(dim):
        basis_vector = np.zeros((1, dim))
        basis_vector[0, i] = 1
        basis.append(basis_vector)
    return basis</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.create_cq_channel"><code class="name flex">
<span>def <span class="ident">create_cq_channel</span></span>(<span>dim, n)</span>
</code></dt>
<dd>
<div class="desc"><p>Creates a random cq-channel with input alphabet size n and output dimension dim
Uses randpsd</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def create_cq_channel(dim, n):
    &#39;&#39;&#39;
    Creates a random cq-channel with input alphabet size n and output dimension dim
    Uses randpsd
    &#39;&#39;&#39;
    channel = []
    for i in range(n):
        channel.append(randpsd(dim))
    return channel</code></pre>
</details>
</dd>
<dt id="quantum_blahut_arimoto.randpsd"><code class="name flex">
<span>def <span class="ident">randpsd</span></span>(<span>n)</span>
</code></dt>
<dd>
<div class="desc"><p>Returns a random real psd matrix of dimension n x n, by first creating a random
square matrix M of dimension n and then returning M @ M^T, which is always psd
after making the trace 1</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def randpsd(n):
    &#39;&#39;&#39;
    Returns a random real psd matrix of dimension n x n, by first creating a random 
    square matrix M of dimension n and then returning M @ M^T, which is always psd
    after making the trace 1
    &#39;&#39;&#39;
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j] = random.random()
    M = M @ (M.T)
    return (1/(np.trace(M))) * M  </code></pre>
</details>
</dd>
</dl>
</section>
<section>
</section>
</article>
<nav id="sidebar">
<h1>Index</h1>
<div class="toc">
<ul></ul>
</div>
<ul id="index">
<li><h3><a href="#header-functions">Functions</a></h3>
<ul class="">
<li><code><a title="quantum_blahut_arimoto.D" href="#quantum_blahut_arimoto.D">D</a></code></li>
<li><code><a title="quantum_blahut_arimoto.F" href="#quantum_blahut_arimoto.F">F</a></code></li>
<li><code><a title="quantum_blahut_arimoto.J" href="#quantum_blahut_arimoto.J">J</a></code></li>
<li><code><a title="quantum_blahut_arimoto.act_channel" href="#quantum_blahut_arimoto.act_channel">act_channel</a></code></li>
<li><code><a title="quantum_blahut_arimoto.adjoint_channel" href="#quantum_blahut_arimoto.adjoint_channel">adjoint_channel</a></code></li>
<li><code><a title="quantum_blahut_arimoto.capacity" href="#quantum_blahut_arimoto.capacity">capacity</a></code></li>
<li><code><a title="quantum_blahut_arimoto.complementary_channel" href="#quantum_blahut_arimoto.complementary_channel">complementary_channel</a></code></li>
<li><code><a title="quantum_blahut_arimoto.create_amplitude_damping_channel" href="#quantum_blahut_arimoto.create_amplitude_damping_channel">create_amplitude_damping_channel</a></code></li>
<li><code><a title="quantum_blahut_arimoto.create_basis" href="#quantum_blahut_arimoto.create_basis">create_basis</a></code></li>
<li><code><a title="quantum_blahut_arimoto.create_cq_channel" href="#quantum_blahut_arimoto.create_cq_channel">create_cq_channel</a></code></li>
<li><code><a title="quantum_blahut_arimoto.randpsd" href="#quantum_blahut_arimoto.randpsd">randpsd</a></code></li>
</ul>
</li>
</ul>
</nav>
</main>
<footer id="footer">
<p>Generated by <a href="https://pdoc3.github.io/pdoc"><cite>pdoc</cite> 0.9.2</a>.</p>
</footer>
</body>
</html>