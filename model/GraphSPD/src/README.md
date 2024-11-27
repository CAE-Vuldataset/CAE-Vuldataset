# Paimon
Paimon: <u>Pa</u>tch <u>I</u>dentification <u>Mon</u>ster
* Notes: This is the instruction of old-version Paimon.

## 1. Dependencies

```bash
$ conda create -n paimon python=3.9
$ conda activate paimon
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ conda install pyg -c pyg
$ conda install transformer
```

## 2. Data Preprocessing

All commands are executed under the root folder: `<PATH_TO_FOLDER>/Paimon/`, which is refered as `<root>` in the following instructions.

### Step 1: Generate raw dataset from Joern results.

```bash
$ python ./src/get_dataset.py
```

This step is to verify, copy, rename, and label the original files. 

According to the labeled patch list in `<root>/_PatchDB/`, we verify if each file in `<root>/_GraphLogs/` is valid, i.e., there is corresponding patch is in PatchDB. 

We copy and rename the valid files to `<root>/data_raw/[positives|negatives]/*.log` and determine the `positives` or `negatives` subfolder based on the PatchDB label.

This step only moves the files and does not change the graph format:

```vim
(nodeid_out, nodeid_in, edge_type, edge_version)
===========================    <-- delimiter between edges and nodes.
(nodeid, node_version, node_type, dist, line_num, code_line)
```

For example, a graph with 2 nodes and 1 edge:

```vim
(-3, -37, 'DDGcbb', 1)                          
===========================
(-3, 0, 'D', 8, '82', 'CBB *cbb')               
(-37, 1, '-', 0, '+90', 'cbb->child = NULL')
```

### Step 2: Parse the graph data from text to numpy object.

```bash
$ python ./src/parse_graphs.py
```

This step is to parse graphs in the text records.

We parse the nodes and edges in the raw text data and then construct the graph data in numpy object format. 

By this step, the graph file in `<root>/data_raw/*/*.log` will be converted into `<root>/data_mid/*/*.npz`.

The mid-point graph data structure can be shown by loading the `.npz` files:

```python
>>> gdata = np.load('gdata_mid.npz', allow_pickle=True)
>>> list(gdata.keys()) # the items regarding a graph.
['nodesP', 'edgesP', 'nodesA', 'edgesA', 'nodesB', 'edgesB', 'label', 'dtype']
# nodes and edges of patch (P), pre-patch (A) code, and post-patch (B) code.
>>> gdata['nodesP']
array([['-3', '0', 'D', '8', '82', 'CBB *cbb'],
       ['-37', '1', '-', '0', '+90', 'cbb->child = NULL']], dtype=object)
# each node contains nodeid, node_version, node_type, dist, line_num, code_tokens.
>>> gdata['edgesP']
array([['-3', '-37', 'DDG', '1']], dtype=object)
# each edge contains nodeid_out, nodeid_in, edge_type, edge_version.
>>> gdata['label']
array([1]) # label.
```

### Step 3: Embed the graph data.

```bash
$ python ./src/embed_graphs.py
```

This step is for ID mapping, tokenization, and embedding.

**(1) ID Mapping**

We map the valid node IDs to the 0-indexed consecutive integers, i.e., from `0`, `1` ... to `#valid_nodes-1`. Here, the valid node means the node exist in the node list and is used in edge connection.

There are two other situations. 

* The node exists in the node list, but is not used in edge connection; it is an isolated point and it will be dropped. The graph will be still processed and saved. 

* The node is used in edge connection, but we cannot find its node id in the node list; the generated graph information is imcompleted, it will raise an error `[Error] <ProcNodes> Node [node_id] does not in node list`. In this situation, the broken graph will be dropped and nothing is saved.

Also, there is two special cases. 

* If the graph is void (i.e., no node and no edge), the graph will be processed as a graph with two nodes connected to each other. All the node/edge features will be zeros with the normal dimensions. This case usally happens in one of the twin graphs.

* If the graph is a N-node graph without any edge, the graph will also be processed as a graph only with two nodes connected to each other. The edge features will be zeros with the normal dimensions. The two node features will come from the first 2 nodes if applicable. If the graph only contains 1 node, both 2 features come from this node.   

**(2) Tokenization**

The tokenization step is processed using `RobertaTokenizer`, while the code statement will be truncated into 512-token length if it is too long.

**(3) Embedding**

The embedding step is processed by `RobertaModel`. you can use `pooler_output` or `last_hidden_state` with aggregation.

This saved graph contains 4 mandatory information: 

* `edgeIndex` is a `2 * #edges` matrix recording the edge connection and graph topology. It is a sparse version of adjency matrix.
* `edgeAttr` is a `#edges * 5` matrix recording the edge embedding.
* `nodeAttr` is a `#nodes * 768` matrix recording the node embedding.
* `label` is a `1 * 1` vector recording the graph label.

We can see the structure of saved patch graph:

```python
>>> g_patch = np.load('gdata_patch.npz', allow_pickle=True)
>>> list(g_patch.keys())
['edgeIndex', 'edgeAttr', 'nodeAttr', 'label', 'nodeDict']
>>> g_patch['edgeIndex']
array([[0],
       [1]])
>>> g_patch['edgeAttr']
array([[0, 1, 0, 1, 0]])
>>> g_patch['nodeAttr']
array([[ 0.3733095 , -0.38387725, -0.4817224 , ..., -0.03157633,
        -0.02150492, -0.0046486 ],
       [ 0.41430753, -0.41784966, -0.5413571 , ..., -0.04296428,
         0.01779909,  0.02394774]], dtype=float32)
>>> g_patch['label']
array([1])
>>> g_patch['nodeDict'] # debug only, no need to use.
array({'-3': 0, '-37': 1}, dtype=object)
```

We also construct a twin graph structure to record the pre-patch and post-patch information. 

The `edgeIndex`, `edgeAttr`, `nodeAttr` uses the `0` and `1` suffix to indicate pre-patch and post-patch info.

```python
>>> g_twin = np.load('gdata_twin.npz', allow_pickle=True)
>>> list(g_twin.keys())
['edgeIndex0', 'edgeAttr0', 'nodeAttr0', 'edgeIndex1', 'edgeAttr1', 'nodeAttr1', 'label']
```

## 3. Model Training.

Perform Patch-GNN:

```bash
python src/gnn_patch.py
```
or Twin-GNN:
```bash
python src/gnn_twin.py
```

## A. Standard Formats

### A.1. Text format of graph.

The parser follows the text format to get the graph information. The file is end with `.log`.

```vim
edge list of patch graph
===========================    <-- delimiter between edges and nodes.
node list of patch graph
---------------------------    <-- delimiter between graphs.
edge list of pre-patch graph
===========================
node list of pre-patch graph
---------------------------
edge list of post-patch graph
===========================
node list of post-patch graph
```

**Important Notes:** Use only ONE blank line when there is no edge or node in the list.


Edge format:

```vim
(-id1, -id2, ['CDG'|'DDG'], [-1|0|1])
```

* `id1` and `id2` are intergers.
* `['CDG'|'DDG']` means can be either `'CDG'` or `'DDG'`.
* `[-1|0|1]` means can be interger `-1`, `0`, or `1`.

Node format:

```vim
(-id, free_num, 'free_str', free_num, 'free_str', 'code_statement')
```

* `id` is an interger.
* `free_num` are reserved intergers that have not been used.
* `'free_str'` are reserved strings that have not been used.
* `'code_statement'` is a string recording the code line.

### A.2. NumPy object format of graph.

The mid-point graph data structure follows the following numpy format. The file is end with `.npz`.

```python
# node information.
nodeid = 8 # can be an integer or a string.
code = 'if (a > 1)' # must be a string.
node1 = [nodeid, code] # len(node)>=2; node[0]->nodeid, node[-1]->code, node[1:-1]->any.
node2 = [14, 'a += 1;']
node3 = [5, 'int *p = NULL;']
nodes = np.array([node1, node2, node3], dtype=object)
# edge information.
edgetype = 'CDG' # must be a string, can be 'CDG', 'DDG', or 'AST'.
version = '1' # must be a string, can be '-1', '0', or '1'.
edge1 = [8, 5, edgetype, version]
edge2 = [14, 8, 'DDG', '0']
edges = np.array([edge1, edge2], dtype=object)
# label information.
label = np.array([1]) # [0]:non-security or [1]:security.
# save as numpy object 
np.savez('commit_name.npz', nodesP=nodes, edgesP=edges, nodesA=nodes, edgesA=edges, nodesB=nodes, edgesB=edges, label=label, dtype=object)
# save nodes/edges into corresponding graph (P:patch, A:pre-patch, B:post-patch)
```

We can see the graph information by reading the `.npz` file.

```python
>>> gdata = np.load('commit_name.npz', allow_pickle=True)
>>> list(gdata.keys()) # the keys in the numpy object.
['nodesP', 'edgesP', 'nodesA', 'edgesA', 'nodesB', 'edgesB', 'label', 'dtype']
>>> gdata['nodesP'] # same structure in nodesA and nodesB.
array([[8, 'if (a > 1)'],
       [14, 'a += 1;'],
       [5, 'int *p = NULL;']], dtype=object)
# each node contains nodeid, codeline.
>>> gdata['edgesP'] # same structure in edgesA and edgesB.
array([[8, 5, 'CDG', '1'],
       [14, 8, 'DDG', '0']], dtype=object)
# each edge contains nodeid_out, nodeid_in, edge_type, edge_version.
>>> gdata['label']
array([1]) # label.
```

After executing `./src/embed_graphs.py`, we can see this graph data structure can be processed.

```python
>>> g_patch = np.load('commit_name.npz', allow_pickle=True)
>>> list(g_patch.keys())
['edgeIndex', 'edgeAttr', 'nodeAttr', 'label', 'nodeDict']
>>> g_patch['edgeIndex']
array([[0, 1],
       [2, 0]])
>>> g_patch['edgeAttr']
array([[0, 1, 1, 0, 0],
       [1, 1, 0, 1, 0]])
>>> g_patch['nodeAttr']
array([[ 0.38118038, -0.42532718, -0.6148339 , ..., -0.03758701,
         0.01594985,  0.08632272],
       [ 0.34696624, -0.45462358, -0.5188762 , ..., -0.04286851,
         0.02313088,  0.04798017],
       [ 0.41223133, -0.45855263, -0.5493048 , ..., -0.08573418,
        -0.01763462,  0.1002164 ]], dtype=float32)
>>> g_patch['label']
array([1])
>>> g_patch['nodeDict']
array({8: 0, 14: 1, 5: 2}, dtype=object)
```