from models import *
from helper import *
import tensorflow as tf

"""
Abbreviations used in variable names:
	et: event-time
	de: dependency parse
"""

class DCT_NN(Model):
	# Pads the data in a batch
	def padData(self, data, seq_len):
		temp = np.zeros((len(data), seq_len), np.int32)
		mask = np.zeros((len(data), seq_len), np.float32)
		
		for i, ele in enumerate(data):
			temp[i, :len(ele)] = ele[:seq_len]
			mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

		return temp, mask

	# Generates the one-hot representation
	def getOneHot(self, data, num_class):
		temp = np.zeros((len(data), num_class), np.int32)
		for i, ele in enumerate(data):
			temp[i, ele] = 1
		return temp

	def getBatches(self, data, shuffle = True):
		if shuffle: random.shuffle(data)
		num_batches = len(data) // self.p.batch_size

		for i in range(num_batches):
			start_idx = i * self.p.batch_size
			yield data[start_idx : start_idx + self.p.batch_size]

	# Merges edge labels or Ignores Edge labels based on cmd arguments
	def updateEdges(self, data, merge_edges=False):

		for dtype in ['train', 'test', 'valid']:
			for i, edges in enumerate(data[dtype]['ETEdges']):
				for j in range(len(edges)-1, -1, -1):
					edge = edges[j]
					lbl  = self.id2ce[edge[2]]
					
					if lbl not in self.n_et2id: del data[dtype]['ETEdges'][i][j]
					else: 			    data[dtype]['ETEdges'][i][j] = (edge[0], edge[1], self.n_et2id[lbl])

			if merge_edges:
				for i, edges in enumerate(data[dtype]['ETEdges']):
					for j, edge in enumerate(edges):
						if   edge[2] == self.n_et2id['BEFORE']: 	data[dtype]['ETEdges'][i][j] = (edge[1], edge[0], self.n_et2id['AFTER'])
						elif edge[2] == self.n_et2id['INCLUDES']: 	data[dtype]['ETEdges'][i][j] = (edge[1], edge[0], self.n_et2id['IS_INCLUDED'])
			
			# Remove dependency edges with negative source/destination ids
			for i, edges in enumerate(data[dtype]['DepEdges']):
				for j in range(len(edges)-1, -1, -1):
					edge = edges[j]
					if edge[0] < 0 or edge[1] < 0:
						del data[dtype]['DepEdges'][i][j]

		if merge_edges: self.num_etLabel -= 2
		return data
	
	# Remove documents with very large number of edges in Event-Time Graph
	def rm_hdeg_docs(self, data):
		rm_idx = {}
		for dtype in ['train', 'test', 'valid']:
			rm_idx[dtype] = set()
			
			for i,vec in enumerate(data[dtype]['ETIdx']):
				if len(vec) > self.p.th_maxet: 
					rm_idx[dtype].add(i)
			
			for i,vec in enumerate(data[dtype]['ET']):
				if len(vec)> self.p.th_seq_len:
					rm_idx[dtype].add(i)

			for i, etIdx in enumerate(data[dtype]['ETIdx']):
				if len(etIdx) == 0: 
					rm_idx[dtype].add(i)
		return rm_idx


	# Loads the data and arranges data for feeding to TensorFlow
	def load_data(self):
		data = pickle.load(open(self.p.dataset, 'rb'))

		self.voc2id 	= data['voc2id']
		self.id2voc 	= data['id2voc']
		self.tense2id 	= data['tense2id']
		self.et2id 	= data['et2id']
		self.id2ce  	= dict([(v,k) for k,v in self.et2id.items()])
		self.de2id 	= data['de2id']

		self.n_et2id = {
			'AFTER': 	0,
			'IS_INCLUDED':	1,
			'SIMULTANEOUS':	2,
			'DURING':	2,
			'BEFORE':	3,
			'INCLUDES':	4,
		}


		self.num_etLabel  = len(self.n_et2id)
		self.num_deLabel  = len(self.de2id)
		data 		  = self.updateEdges(data, self.p.merge_edges)			# Merge edge labels
		rm_idx 		  = self.rm_hdeg_docs(data)					# Indexes to be removed

		print('Number of classes {}'.format(len(np.unique(data['train']['Y']))))
		self.num_class = self.p.num_class

		self.logger.info('Removing Train:{}, Test:{}, Valid:{}'.format(len(rm_idx['train']), len(rm_idx['test']), len(rm_idx['valid'])))

		# Get Word List
		self.wrd_list 	= list(self.voc2id.items())					# Get vocabulary
		self.wrd_list.sort(key=lambda x: x[1])						# Sort vocabulary based on ids
		self.wrd_list, _ = zip(*self.wrd_list)
		
		self.data_list = {}
		key_list =  ['X', 'Y', 'ETIdx', 'ETEdges', 'DepEdges', 'Fname']

		for dtype in ['train', 'test', 'valid']:

			if self.p.use_et_labels == False:
				for i, edges in enumerate(data[dtype]['ETEdges']):												# if you want to ignore level information in event time graph
					for j, edge in enumerate(edges): data[dtype]['ETEdges'][i][j] = (edge[0], edge[1], 0)   
				self.num_etLabel = 1

			if self.p.use_de_labels == False:
				for i, edges in enumerate(data[dtype]['DepEdges']):												# if you want to ignore level information in dependency graph
					for j, edge in enumerate(edges): data[dtype]['DepEdges'][i][j] = (edge[0], edge[1], 0)
				self.num_deLabel = 1

			data[dtype]['Y'] = self.getOneHot(data[dtype]['Y'], self.num_class)		# Representing labels by one hot notation

			self.data_list[dtype] = []
			for i in range(len(data[dtype]['X'])):
				if i in rm_idx[dtype]: continue
				self.data_list[dtype].append([data[dtype][key][i] for key in key_list])          # data_list contains all the fields for train test and valid documents

			self.logger.info('Document count [{}]: {}'.format(dtype, len(self.data_list[dtype])))
		self.Et_index = data['valid']['ETIdx']
		self.data = data

	# Loads adjacency matrix in sparse matrix format, required for feeding to Tensorflow
	def get_adj(self, edgeList, batch_size, max_nodes, max_labels):
		adj_main_in, adj_main_out = [], []

		for edges in edgeList:
			adj_in, adj_out = {}, {}

			in_ind, in_data   = ddict(list), ddict(list)
			out_ind, out_data = ddict(list), ddict(list)

			for src, dest, lbl in edges:
				out_ind [lbl].append((src, dest))
				out_data[lbl].append(1.0)

				in_ind  [lbl].append((dest, src))
				in_data [lbl].append(1.0)
			try:
				for lbl in range(max_labels):
					if lbl not in out_ind and lbl not in in_ind:
						adj_in [lbl] = sp.coo_matrix((max_nodes, max_nodes))
						adj_out[lbl] = sp.coo_matrix((max_nodes, max_nodes))
					else:
						adj_in [lbl] = sp.coo_matrix((in_data[lbl],  zip(*in_ind[lbl])),  shape=(max_nodes, max_nodes))
						adj_out[lbl] = sp.coo_matrix((out_data[lbl], zip(*out_ind[lbl])), shape=(max_nodes, max_nodes))
			except Exception as e:
				pdb.set_trace()

			adj_main_in.append(adj_in)
			adj_main_out.append(adj_out)

		return adj_main_in, adj_main_out


	def add_placeholders(self):
		self.input_x  		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_data')			# Words in a document (batch_size x max_words)
		self.input_y 		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_labels')			# Actual document creation year of the document

		self.x_len		= tf.placeholder(tf.int32,   shape=[None],       name='input_len')					# Number of words in each document in a batch
		self.et_idx 		= tf.placeholder(tf.int32,   shape=[None, None],    name='et_idx')				# Index of tokens which are events/time_expressions
		self.et_mask 		= tf.placeholder(tf.float32, shape=[None, None],    name='et_mask')

		# Array of batch_size number of dictionaries, where each dictionary is mapping of label to sparse_placeholder [Temporal graph]
		self.et_adj_mat_in	= [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'et_adj_mat_in_{}'. format(lbl))) for lbl in range(self.num_etLabel)]) for i in  range(self.p.batch_size) ]
		self.et_adj_mat_out 	= [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'et_adj_mat_out_{}'.format(lbl))) for lbl in range(self.num_etLabel)]) for i in  range(self.p.batch_size) ]

		# Array of batch_size number of dictionaries, where each dictionary is mapping of label to sparse_placeholder [Syntactic graph]
		self.de_adj_mat_in	= [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'de_adj_mat_in_{}'. format(lbl))) for lbl in range(self.num_deLabel)]) for i in  range(self.p.batch_size) ]
		self.de_adj_mat_out 	= [dict([(lbl, tf.sparse_placeholder(tf.float32,  shape=[None, None],  name= 'de_adj_mat_out_{}'.format(lbl))) for lbl in range(self.num_deLabel)]) for i in  range(self.p.batch_size) ]

		self.seq_len 		= tf.placeholder(tf.int32, shape=(), name='seq_len')				# Maximum number of words in documents of a batch
		self.max_et 		= tf.placeholder(tf.int32, shape=(), name='max_et')					# Maximum number of events/time_expressions in documents of a batch

		self.dropout 		= tf.placeholder_with_default(self.p.dropout, 	  shape=(), name='dropout')		# Dropout used in GCN Layer
		self.rec_dropout 	= tf.placeholder_with_default(self.p.rec_dropout, shape=(), name='rec_dropout')	# Dropout used in Bi-LSTM

		self.de_out_mask 	= tf.placeholder(tf.int32,   shape=[None],       name='input_len')

	def pad_dynamic(self, X, et_idx):
		seq_len, max_et, de_out_mask = 0, 0, []

		x_len = np.zeros((len(X)), np.int32)

		for i, x in enumerate(X): 	  
			seq_len  = max(seq_len, len(x))
			x_len[i] = len(x)

		for et in et_idx: max_et  = max(max_et,  len(et))

		x_pad,  _ 	= self.padData(X, seq_len)
		et_pad, et_mask = self.padData(et_idx, max_et)
		return x_pad, x_len, et_pad, et_mask, seq_len, max_et

	def create_feed_dict(self, batch, wLabels=True, dtype='train'):
		X, Y, et_idx, ETEdges, DepEdges, _ = zip(*batch)

		x_pad, x_len, et_pad, et_mask, seq_len, max_et = self.pad_dynamic(X, et_idx)

		feed_dict = {}
		feed_dict[self.input_x] 		= np.array(x_pad)
		feed_dict[self.x_len] 			= np.array(x_len)
		if wLabels: feed_dict[self.input_y] 	= np.array(Y)

		feed_dict[self.et_idx] 			= np.array(et_pad)
		feed_dict[self.et_mask] 		= np.array(et_mask)

		feed_dict[self.seq_len]			= seq_len
		feed_dict[self.max_et]			= max_et

		et_adj_in, et_adj_out = self.get_adj(ETEdges,  self.p.batch_size, max_et+1,  self.num_etLabel)  # max_et + 1(DCT)
		de_adj_in, de_adj_out = self.get_adj(DepEdges, self.p.batch_size, seq_len, self.num_deLabel)

		for i in range(self.p.batch_size):
			for lbl in range(self.num_etLabel):
				feed_dict[self.et_adj_mat_in[i][lbl]] = tf.SparseTensorValue( 	indices 	= np.array([et_adj_in[i][lbl].row, et_adj_in[i][lbl].col]).T,
											      								values  	= et_adj_in[i][lbl].data,
																				dense_shape	= et_adj_in[i][lbl].shape)

				
				feed_dict[self.et_adj_mat_out[i][lbl]] = tf.SparseTensorValue(  indices 	= np.array([et_adj_out[i][lbl].row, et_adj_out[i][lbl].col]).T,
				    								values  	= et_adj_out[i][lbl].data,
		 										dense_shape	= et_adj_out[i][lbl].shape)

			for lbl in range(self.num_deLabel):
				feed_dict[self.de_adj_mat_in[i][lbl]] = tf.SparseTensorValue( 	indices 	= np.array([de_adj_in[i][lbl].row, de_adj_in[i][lbl].col]).T,
											      	values  	= de_adj_in[i][lbl].data,
												dense_shape	= de_adj_in[i][lbl].shape)

				feed_dict[self.de_adj_mat_out[i][lbl]] = tf.SparseTensorValue(  indices 	= np.array([de_adj_out[i][lbl].row, de_adj_out[i][lbl].col]).T,
				    								values  	= de_adj_out[i][lbl].data,
		 		 									dense_shape	= de_adj_out[i][lbl].shape)
		feed_dict[self.de_out_mask] = np.array(x_len)
		if dtype != 'train':
			feed_dict[self.dropout]     = 1.0
			feed_dict[self.rec_dropout] = 1.0

		return feed_dict


	# GCN Layer Implementation for S-GCN
	def gcnLayer(self, gcn_in,		# Input to GCN Layer
					 in_dim, 		# Dimension of input to GCN Layer 
					 gcn_dim, 		# Hidden state dimension of GCN
					 batch_size, 	# Batch size
					 max_nodes, 	# Maximum number of nodes in graph
					 max_labels, 	# Maximum number of edge labels
					 adj_in, 		# Adjacency matrix for in edges
					 adj_out, 		# Adjacency matrix for out edges
					 num_layers=1, 	# Number of GCN Layers
					 name="GCN"):
		out = []
		out.append(gcn_in)

		for layer in range(num_layers):
			gcn_in    = out[-1]								# out contains the output of all the GCN layers, intitally contains input to first GCN Layer
			if len(out) > 1: in_dim = gcn_dim 				# After first iteration the in_dim = gcn_dim

			with tf.name_scope('%s-%d' % (name,layer)):

				act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
				
				for lbl in range(max_labels):

					with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:

						w_in   = tf.get_variable('w_in',   [in_dim, gcn_dim],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						b_in   = tf.get_variable('b_in',   [1, gcn_dim],   	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)

						w_out  = tf.get_variable('w_out',  [in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						b_out  = tf.get_variable('b_out',  [1, gcn_dim],  	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)

						w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)

						if self.p.wGate:
							w_gin  = tf.get_variable('w_gin',  [in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
							b_gin  = tf.get_variable('b_gin',  [1], 	  	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)

							w_gout = tf.get_variable('w_gout', [in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
							b_gout = tf.get_variable('b_gout', [1], 	  	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)

							w_gloop = tf.get_variable('w_gloop',[in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)

					with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)
						in_t    = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], inp_in[i]) for i in range(batch_size)])
						if self.p.dropout != 1.0: in_t    = tf.nn.dropout(in_t, keep_prob=self.p.dropout)

						if self.p.wGate:
							inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)
							in_gate = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], inp_gin[i]) for i in range(batch_size)])
							in_gsig = tf.sigmoid(in_gate)
							in_act   = in_t * in_gsig
						else:
							in_act   = in_t

					with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
						out_t    = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], inp_out[i]) for i in range(batch_size)])
						if self.p.dropout != 1.0: out_t    = tf.nn.dropout(out_t, keep_prob=self.p.dropout)

						if self.p.wGate:
							inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
							out_gate = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], inp_gout[i]) for i in range(batch_size)])
							out_gsig = tf.sigmoid(out_gate)
							out_act  = out_t * out_gsig
						else:
							out_act = out_t

					with tf.name_scope('self_loop'):
						inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
						if self.p.dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

						if self.p.wGate:
							inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2,0])
							loop_gsig = tf.sigmoid(inp_gloop)
							loop_act  = inp_loop * loop_gsig
						else:
							loop_act = inp_loop

					act_sum += in_act + out_act + loop_act


				gcn_out = tf.nn.relu(act_sum)
				out.append(gcn_out)

		return out


	# GCN Layer Implementation for AT-GCN
	def gcnLayerDCT(self, gcn_in, # Input to GCN Layer
					 in_dim, 		# Dimension of input to GCN Layer 
					 gcn_dim, 		# Hidden state dimension of GCN
					 batch_size, 	# Batch size
					 max_nodes, 	# Maximum number of nodes in graph
					 max_labels, 	# Maximum number of edge labels
					 adj_in, 		# Adjacency matrix for in edges
					 adj_out, 		# Adjacency matrix for out edges
					 num_layers=1, 	# Number of GCN Layers
					 name="GCN"):
		out = []
		keep_attn_in = tf.zeros([batch_size, max_nodes, max_nodes])
		keep_attn_out = tf.zeros([batch_size, max_nodes, max_nodes])
		out.append(gcn_in)
		keep_gcn = gcn_in
		keep_max_node = max_nodes
		for layer in range(num_layers):
			gcn_in    = out[-1]								# out contains the output of all the GCN layers, intitally contains input to first GCN Layer
			if len(out) > 1: in_dim = gcn_dim 				# After first iteration the in_dim = gcn_dim

			with tf.name_scope('%s-%d' % (name,layer)):

				act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])

				
				for lbl in range(max_labels):

					with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:

						w_in   = tf.get_variable('w_in',   [in_dim, gcn_dim],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						b_in   = tf.get_variable('b_in',   [1, gcn_dim],   	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)

						w_out  = tf.get_variable('w_out',  [in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						b_out  = tf.get_variable('b_out',  [1, gcn_dim],  	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)

						w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)

					with tf.variable_scope('name-%s_layer-%d' % (name, layer), reuse = tf.AUTO_REUSE) as scope:
						a_in_1 = tf.get_variable('a_in_1',   [gcn_dim, 1],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						a_in_2 = tf.get_variable('a_in_2',   [gcn_dim, 1],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						a_out_1 = tf.get_variable('a_out_1',   [gcn_dim, 1],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						a_out_2 = tf.get_variable('a_out_2',   [gcn_dim, 1],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
					with tf.variable_scope('name-%s_layer-%d' % (name, layer), reuse = tf.AUTO_REUSE) as scope:
						w_attn_in = tf.get_variable('w_attn_in',   [in_dim, in_dim],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						w_attn_out = tf.get_variable('w_attn_out',  [in_dim, in_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)



					with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)
						attn_temp_in = tf.tanh(tf.tensordot(inp_in, w_attn_in, axes=[2,0]))
						attn_temp_in = tf.reshape(attn_temp_in, [-1,gcn_dim])
						if self.p.dropout != 1.0: attn_temp_in    = tf.nn.dropout(attn_temp_in, keep_prob=self.p.dropout)
						attn_mid_1_in = tf.tile(tf.expand_dims(tf.reshape(tf.matmul(attn_temp_in, a_in_1), [self.p.batch_size, -1]), -1), [1, 1, max_nodes])
						attn_mid_2_in = tf.tile(tf.transpose(tf.expand_dims(tf.reshape(tf.matmul(attn_temp_in, a_in_2), [self.p.batch_size, -1]), -1), [0, 2, 1]), [1, max_nodes, 1])
						attn_in_val = attn_mid_1_in + attn_mid_2_in
						attn_in = attn_in_val
						attn_in = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], tf.eye(max_nodes, max_nodes))*attn_in[i] for i in range(batch_size)])
						attn_in = tf.exp(attn_in)
						attn_in = tf.stack([tf.sparse_tensor_dense_matmul(adj_in[i][lbl], tf.eye(max_nodes, max_nodes))*attn_in[i] for i in range(batch_size)])
						keep_attn_in_val = []
						for i in range(batch_size):
							attn_temp = attn_in[i]
							keep_attn_in_val.append(attn_temp / (tf.reshape(tf.reduce_sum(attn_temp, axis=1) + tf.constant(0.00000001), (-1, 1))))
						attn_in_final = tf.stack(keep_attn_in_val)


						in_act    = tf.matmul(attn_in_final, inp_in)
						if self.p.dropout != 1.0: in_act    = tf.nn.dropout(in_act, keep_prob=self.p.dropout)

					with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
						attn_temp_out = tf.tanh(tf.tensordot(inp_out, w_attn_out, axes=[2,0]))
						attn_temp_out = tf.reshape(attn_temp_out, [-1,gcn_dim])
						if self.p.dropout != 1.0: attn_temp_out    = tf.nn.dropout(attn_temp_out, keep_prob=self.p.dropout)
						attn_mid_1_out = tf.tile(tf.expand_dims(tf.reshape(tf.matmul(attn_temp_out, a_out_1), [self.p.batch_size, -1]), -1), [1, 1, max_nodes])
						attn_mid_2_out = tf.tile(tf.transpose(tf.expand_dims(tf.reshape(tf.matmul(attn_temp_out, a_out_2), [self.p.batch_size, -1]), -1), [0, 2, 1]), [1, max_nodes, 1])
						attn_out_val = attn_mid_1_out + attn_mid_2_out
						attn_out = attn_out_val
						attn_out = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], tf.eye(max_nodes, max_nodes))*attn_out[i] for i in range(batch_size)])
						attn_out = tf.exp(attn_out)
						attn_out = tf.stack([tf.sparse_tensor_dense_matmul(adj_out[i][lbl], tf.eye(max_nodes, max_nodes))*attn_out[i] for i in range(batch_size)])
						keep_attn_out_val = []
						for i in range(batch_size):
							attn_temp = attn_out[i]
							keep_attn_out_val.append(attn_temp / (tf.reshape(tf.reduce_sum(attn_temp, axis=1) + tf.constant(0.00000001), (-1, 1))))
						attn_out_final = tf.stack(keep_attn_out_val)


						out_act    = tf.matmul(attn_out_final, inp_in)
						if self.p.dropout != 1.0: out_act    = tf.nn.dropout(out_act, keep_prob=self.p.dropout)


					with tf.name_scope('self_loop'):
						inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
						if self.p.dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

						loop_act  = inp_loop
					if layer == num_layers-1:
						keep_attn_in = tf.stack([keep_attn_in[i]+attn_in_final[i] for i in range(batch_size)])
						keep_attn_out = tf.stack([keep_attn_out[i]+attn_out_final[i] for i in range(batch_size)])

					act_sum += in_act + out_act + loop_act


				gcn_out = tf.nn.relu(act_sum)
				out.append(gcn_out)
		

		return out, keep_attn_in, keep_attn_out

	# Lookup equivalent for tensors with dim > 2 
	def gather(self, data, pl_idx, pl_mask, max_len, name=None):
		with tf.name_scope(name):
			idx1  = tf.range(self.p.batch_size, dtype=tf.int32)
			idx1  = tf.reshape(idx1, [-1, 1])
			idx1_ = tf.reshape(tf.tile(idx1, [1, max_len]) , [-1, 1])
			idx_reshape = tf.reshape(pl_idx, [-1, 1])
			indices = tf.concat((idx1_, idx_reshape), axis=1)
			et_vecs = tf.gather_nd(data, indices)
			et_vecs = tf.reshape(et_vecs, [self.p.batch_size, self.max_et, -1])
			mask_vec = tf.expand_dims(pl_mask, axis=2)
			temp = et_vecs * mask_vec
			return et_vecs * mask_vec

	# Creates the compuational graph
	def add_model(self):
		nn_in = self.input_x
		with tf.variable_scope('Embeddings') as scope:
			embed_init = getGlove(self.wrd_list, self.p.embed_init)	
			embed_init = np.vstack( (np.zeros(self.p.embed_dim, np.float32), embed_init))
			embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True, regularizer=self.regularizer)
			embeds     = tf.nn.embedding_lookup(embeddings, self.input_x)
		with tf.variable_scope('Bi-LSTM') as scope:
			fw_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.p.lstm_dim), output_keep_prob=self.rec_dropout)
			bk_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.p.lstm_dim), output_keep_prob=self.rec_dropout)
			val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embeds, sequence_length=self.x_len, dtype=tf.float32)
			lstm_out   = tf.concat((val[0], val[1]), axis=2)

			de_in = lstm_out
			de_in_dim = self.p.lstm_dim*2		# Concatenated output of forward and backward LSTM (Bi-LSTM)
		

		de_out = self.gcnLayer( gcn_in 		= de_in, 		in_dim 	    = de_in_dim, 		gcn_dim    = self.p.de_gcn_dim, 
					batch_size 	= self.p.batch_size, 	max_nodes   = self.seq_len, 		max_labels = self.num_deLabel, 
					adj_in 	= self.de_adj_mat_in, 		adj_out     = self.de_adj_mat_out, 
						num_layers 	= self.p.de_layers, 	name 	   = "GCN_DE")

		ce_in_dim = self.p.de_gcn_dim
		ce_in = de_out[-1]				# GCNLayer returns list containing output of all layers; last entry is its final output
		

		et_vecs = self.gather(ce_in, self.et_idx, self.et_mask, self.max_et, name='ET_pick')
		with tf.name_scope('DCT_init'):
			dct_sum  = tf.reduce_sum(et_vecs, axis=1)
			dct_cnt  = tf.reduce_sum(self.et_mask, axis=1)
			dct_init = tf.expand_dims(dct_sum / tf.expand_dims(dct_cnt,axis=1), axis=1)

		et_con = tf.concat( [dct_init, et_vecs], axis=1)
		ce_out, store_attn_in, store_attn_out = self.gcnLayerDCT( gcn_in 		= et_con, 		in_dim 		= ce_in_dim, 			gcn_dim    = self.p.et_gcn_dim, 
					batch_size 	= self.p.batch_size, 	max_nodes 	= self.max_et+1, 		max_labels = self.num_etLabel, 			# max_et + 1(DCT)
					adj_in 	= self.et_adj_mat_in, 		adj_out     	= self.et_adj_mat_out,
					num_layers	= self.p.et_layers, 	name 		= "GCN_CE")									

		dct_vec = ce_out[-1][:,0]

		dct_final = dct_vec
		fc_in_dim = self.p.et_gcn_dim

		with tf.variable_scope('FC1') as scope:
			w = tf.get_variable('w', [fc_in_dim, self.num_class], 	initializer=tf.truncated_normal_initializer(),  regularizer=self.regularizer)
			b = tf.get_variable('b', [self.num_class], 	  	initializer=tf.constant_initializer(0.0), 	regularizer=self.regularizer)
			nn_out = tf.matmul(dct_final, w) + b
			
		return nn_out, nn_in, store_attn_in, store_attn_out


	def add_loss(self, nn_out):
		with tf.name_scope('Loss_op'):
			loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		return loss

	def add_optimizer(self, loss):
		with tf.name_scope('Optimizer'):
			optimizer = tf.train.AdamOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		return train_op

	def __init__(self, params):
		self.p  = params
		self.logger = get_logger(self.p.name.replace('/', '_'))
		self.logger.info(vars(self.p))
		self.p.batch_size = self.p.batch_size

		if self.p.l2 == 0.0: 	self.regularizer = None
		else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

		self.load_data()
		self.add_placeholders()
		
		nn_out, nn_in, attn_in_store, attn_out_store = self.add_model()

		self.loss      	= self.add_loss(nn_out)   #Compute loss
		self.train_op  	= self.add_optimizer(self.loss)   #Parameter Update
		self.logits 	= tf.nn.softmax(nn_out)	
		self.nn_inp     = nn_in
		self.keep_attn_in     = attn_in_store
		self.keep_attn_out     = attn_out_store

		y_pred 	  = tf.argmax(self.logits, 1)		# Predictions by the model
		corr_pred = tf.equal(tf.argmax(self.input_y, 1), y_pred)
		self.corr_pred = tf.reduce_sum(tf.cast(corr_pred, 'int32'))

		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None

	def predict(self, sess, data, wLabels=True, shuffle=False):
		losses, results, y_pred, y, fnames, logit_list, inps, attn_input, attn_output = [], [], [], [], [], [], [], [], []
		total_correct, total_cnt = 0, 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			
			if not wLabels:
				feed   = self.create_feed_dict(batch, wLabels, dtype='test')
				logits, correct, nninp, attnin, attnout = sess.run([self.logits, self.corr_pred, self.nn_inp, self.keep_attn_in, self.keep_attn_out] , feed_dict = feed)
			else:
				feed  = self.create_feed_dict(batch, dtype='test')
				loss, logits, correct,  nninp, attnin, attnout = sess.run([self.loss, self.logits, self.corr_pred, self.nn_inp, self.keep_attn_in, self.keep_attn_out], feed_dict = feed)
				losses.append(loss)

			total_correct += correct
			total_cnt += len(batch)

			pred_ind    = logits.argmax(axis=1)
			logit_list += logits.tolist()
			y_pred   += pred_ind.tolist()
			inps += nninp.tolist()
			attn_input += attnin.tolist()
			attn_output += attnout.tolist()
			_, Y, _, _, _ ,fname= zip(*batch)
			y += np.array(Y).argmax(axis=1).tolist()
			fnames += list(fname)
			results.append(pred_ind)

			if step % 5 == 0:
				self.logger.info('Evaluating Test/Valid ({}/{}):\t{:.5}\t{:.5}\t{}'.format(step, len(data)//self.p.batch_size, total_correct/total_cnt, np.mean(losses), self.p.name.replace('/', '_')))


		accuracy = float(total_correct)/total_cnt * 100.0

		self.logger.info('Accuracy: {}'.format(accuracy))

		if wLabels: 	return np.mean(losses), results, accuracy, y, y_pred, fnames, logit_list, inps, attn_input, attn_output
		else: 		return 0, results, accuracy, y, y_pred, fnames, logit_list, inps, attn_input, attn_output

	def run_epoch(self, sess, data, epoch, shuffle=True):
		drop_rate = self.p.dropout

		losses = []
		total_correct, total_cnt = 0, 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			loss, correct, _= sess.run([self.loss, self.corr_pred, self.train_op], feed_dict=feed)
			if(np.isnan(loss)): 
				print(et_cnt)
				pdb.set_trace()
			total_cnt     += len(batch)
			total_correct += correct

			losses.append(loss)

			if step % 5 == 0:
				self.logger.info('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, step, len(data)//self.p.batch_size, total_correct/total_cnt, np.mean(losses), self.p.name.replace('/', '_'), self.best_val_acc))

		accuracy = float(total_correct)/total_cnt * 100.0

		self.logger.info('Training Loss:{}, Accuracy: {}'.format(np.mean(losses), accuracy))
		return np.mean(losses), accuracy

	def fit(self, sess):
		self.best_val_acc, self.best_train_acc = 0.0, 0.0
		
		saver = tf.train.Saver()
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		save_path = os.path.join(save_dir, 'best_validation')

		if self.p.restore: saver.restore(sess, save_path)

		self.best_prf = None

		if not self.p.onlyTest:
			for epoch in range(self.p.max_epochs):
				self.logger.info('Epoch: {}'.format(epoch))

				train_loss, train_acc 					   = self.run_epoch(sess,  self.data_list['train'], epoch)
				val_loss, val_pred, val_acc, y, y_pred, fnames, logit_list, _, _, _ = self.predict(sess,	self.data_list['valid'])

				if val_acc > self.best_val_acc:
					self.best_val_acc   = val_acc
					self.best_train_acc = train_acc
					self.best_prf = precision_recall_fscore_support(y, y_pred, average='weighted')
					saver.save(sess=sess, save_path=save_path)

				self.logger.info('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}, Valid Loss: {:.5}, Valid Acc: {:.5} Best Acc: {:.5}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc, self.best_val_acc))
				self.logger.info(self.best_prf)

				try:
					self.log_db.update({'_id': self.p.name.replace('/', '_')}, {
						'$push': {
							"Train_loss": 	float(train_loss),
							"Train_acc": 	float(train_acc), 
							"Valid_loss": 	float(val_loss),
							"Valid_acc":	float(val_acc)
						},
						'$set': {
							"Best_val_acc":		float(self.best_val_acc),
							"Best_train_acc":	float(self.best_train_acc),
							"y_actual":		y,
							"y_pred":		y_pred,
							"results":		list(self.best_prf),
							"fnames":		fnames,
							"Params":		vars(self.p)
						}
					}, upsert=True)
					
				except Exception as e:
					exc_type, exc_obj, exc_tb = sys.exc_info()
					fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
					self.logger.info('\nException Type: {}, \nCause: {}, \nfname: {}, \nline_no: {}'.format(exc_type, e.args[0], fname, exc_tb.tb_lineno))

		self.logger.info('Running on Test set')
		_, test_pred, test_acc, y, y_pred, fnames, logit_list,inp, attn_inputs, attn_outputs = self.predict(sess, self.data_list[self.p.split])
		self.logger.info('Test Acc:{}'.format(test_acc))
		perf = precision_recall_fscore_support(y, y_pred, average='weighted')
		
		word_list = []
		for i in range(len(inp)):
			keep_word = []
			for j in range(len(inp[i])):
				if inp[i][j] == 0:
					break
				keep_word.append(self.wrd_list[inp[i][j]])				
			word_list.append(keep_word)
		res ={
			'Et_index':     self.Et_index,
			'attn_in':		attn_inputs,
			'attn_out':		attn_outputs,	
			'input'  :      word_list,
			'logits' :		logit_list,
			'lbl2id' :		self.data['lbl2id'],
			'y_actual':		y,
			'y_pred':		y_pred,
			'results':		list(perf),
			'fnames':		fnames
		}
		
		self.log_db.update({'_id': self.p.name.replace('/', '_')}, {
			'$set': {
				"y_actual":		y,
				"y_pred":		y_pred,
				"results":		list(perf),
				"fnames":		fnames
			}
		}, upsert=True)

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Main Neural Network for Time Stamping Documents')

	parser.add_argument('-data', 	 dest="dataset", 	required=True,			help='Dataset to use')
	parser.add_argument('-class',	 dest="num_class", 	required=True,   type=int, 	help='Number of classes (years/months)')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='0',			help='GPU to use')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),help='Name of the run')
	parser.add_argument('-embed', 	 dest="embed_init", 	default='wiki_300',	 	help='Embedding for initialization')
	parser.add_argument('-drop',	 dest="dropout", 	default=1.0,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-drop_half',	 dest="drop_half", 	action = 'store_true', help='Use dropout for half epochs')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=1.0,  	type=float,	help='Recurrent dropout for LSTM')
	parser.add_argument('-lr',	 dest="lr", 		default=0.001,  type=float,	help='Learning rate')
	parser.add_argument('-attn', 	 dest="attn", 	action='store_true', 	help='Use attention for CE')
	parser.add_argument('-batch', 	 dest="batch_size", 	default=64,   	type=int, 	help='Batch size')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default=50,   	type=int, 	help='Max epochs')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.001, 	type=float, 	help='L2 regularization')
	parser.add_argument('-l2_half', 	 dest="l2_half", 	action='store_true', 	help='use l2 for half epochs')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization')
	parser.add_argument('-lstm_dim', dest="lstm_dim", 	default=128,   	type=int, 	help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-de_dim',   dest="de_gcn_dim", 	default=128,   	type=int, 	help='Hidden state dimension of GCN over dependency tree')
	parser.add_argument('-et_dim',   dest="et_gcn_dim", 	default=128,   	type=int, 	help='Hidden state dimension of GCN over ET-graphs')
	parser.add_argument('-fc1_dim',  dest="fc1_dim", 	default=128,   	type=int, 	help='Hidden state dimension of FC layer')
	parser.add_argument('-de_layer', dest="de_layers", 	default=1,   	type=int, 	help='Number of layers in GCN over dependency tree')
	parser.add_argument('-et_layer', dest="et_layers", 	default=2,   	type=int, 	help='Number of layers in GCN over ET-graph')
	parser.add_argument('-logdb', 	 dest="log_db", 	default='mod_run',	 	help='MongoDB database for dumping results')
	parser.add_argument('-DE', 	 dest="DE", 		default='gcn', 			choices=['gated', 'plain', 'gcn', 'none'], help='Use DE just for enchancing time/event embedings')
	parser.add_argument('-noGate', 	 dest="wGate", 		action='store_false', 		help='Use gating in GCN')
	parser.add_argument('-split', 	 dest="split", 		default='valid', 		help='Split to use for evaluation')
	parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true', 		help='Evaluate model on test')
	parser.add_argument('-wETmean',	 dest="wETmean", 	action='store_true', 		help='Include ET mean in final representation')
	parser.add_argument('-wAttn',	 dest="wAttn", 	action='store_true', 		help='Use attention or not')	
	parser.add_argument('-merge', 	 dest="merge_edges", 	action='store_true', 		help='Merge edge labels in ET-graph')
	parser.add_argument('-de_lbl', 	 dest="use_de_labels", 	action='store_true', 		help='Use edge labels in dependency tree')
	parser.add_argument('-no-et_lbl',dest="use_et_labels", 	action='store_false', 		help='Ignore edge labels in ET-graph')
	parser.add_argument('-fix_emb',	 dest="fix_emb",	action='store_true',		help='fix embedding for fast training')
	parser.add_argument('-dct', 	 dest="dct_type", 	default='avg', choices=['concat', 'avg', 'last'], 	help='Select the method for constructing embedding for DCT node')
	parser.add_argument('-lstm',	 dest="wLSTM", 		action='store_true', 		help='Include Bi-LSTM in model')
	parser.add_argument('-Cont',	 dest="wCE", 		action='store_false', 		help='With or without ET graph')
	parser.add_argument('-th_et',	 dest="th_maxet", 	default=300 , 	type=int,	help='maximum et_nodes')
	parser.add_argument('-th_seq',	 dest="th_seq_len", 	default=800 , 	type=int,	help='maximum de_nodes or sequence_length')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 		help='Restore from the previous best saved model')
	parser.add_argument('-logdir',	 dest="log_dir", 	default='./log/', 		help='Log directory')
	args = parser.parse_args()

	args.embed_dim = int(args.embed_init.split('_')[1])
	if not args.restore: args.name = args.name + '__' + time.strftime("%d/%m/%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model  = DCT_NN(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)