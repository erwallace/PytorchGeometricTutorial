# Lecture 1

All graphs can be represented as an adjacency matrix, so why not train on these:
- Adj is not invariant to node ordering (i.e. if you swap the label of two node the grpah is remmains the same but Adj does not.)
- Scales with N^2 with number of nodes.

Graph $G = (V, E)$ is made of nodes ($V$) and edges ($E$).

$A = Adj(G) \in \mathbb{R}^{|V| \times |V|}$

$X \in \mathbb{R}^{m \times |V|}$  # X contains node features (each row is a single node)

**Computation Graph**: the neighbours of a node defines its computation graph. This is acted upon by an *order invariant* operation e.g. sum, average. This is how the imputed values of a node are updated.

$$h_v^{(0)} = X_v$$

The representation $h$ of node $v$ in layer 0 (i.e. tthe 0th embedding) is its original features, row $v$ of matrix $X$ ($X_v$)

$$Z_v = h_v^{(K)}$$

The embedding $Z$ of node $v$ is the representation after the final step $K$

h is updated through the following mechanism:

$$h_v^{(k+1)} = \sigma\left(W_k \sum_{u \in N(v)} \frac{h_u^{(k)}}{|N(v)|} + B_k h_v^{(k)}\right)$$

where $\sigma$ is an activateion function (e.g. ReLu), $u$ are the neighbours of $v$. $B$ and $W$ are tunable weights and biases. $h_v$ is updated with the *average* (order invariant operation) embedding of it's neighbours.

**Graph SAGE**


This update function can be re-written as:

$$h_v^{(k+1)} = \sigma\left([W_k  AGG( h_u^{(k)} ), B_k h_v^{(k)}]\right)$$

where $AGG()$ is any order invariant operation. And instead of adding terms together ($W + B$), they concatenate them.



