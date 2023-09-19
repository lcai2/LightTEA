# import all the requirements
import faiss 
from utils import *
from alignment import *
import tensorflow as tf
import tensorflow.keras.backend as K
import time
from args import args
print(args)

#choose the GPU, "-1" represents using the CPU

import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

# choose the dataset and set the random seed
# the first run may be slow because the graph needs to be preprocessed into binary cache
main_st = time.time()
np.random.seed(12306)

dataset = args.dataset
path = "./data/"+ dataset +"/"

# set hyper-parameters, load graphs and pre-aligned entity pairs
# if your GPU is out of memory, try to reduce the ent_dim

seed_number = args.seed_num 
ent_dim = args.dim
depth = args.depth
top_k = args.top_k
alpha = args.alpha
beta = args.beta

if "ICEWS" in dataset: 
    rel_dim, mini_dim = ent_dim//2, 16
    time_dim = ent_dim//2   
    if (seed_number == 1000):
        simt = np.load("./data/ICEWS05-15/simt/simt_ICEWS05-15_1000.npy")
    else:
        simt = np.load("./data/ICEWS05-15/simt/simt_ICEWS05-15_200.npy")
else:
    rel_dim, mini_dim = ent_dim//3, 16
    time_dim = ent_dim//3
    if (seed_number == 1000):
        simt = np.load("./data/YAGO-WIKI50K/simt/simt_YAGO-WIKI50K_1000.npy")
    else:
        simt = np.load("./data/YAGO-WIKI50K/simt/simt_YAGO-WIKI50K_5000.npy")
    
node_size, rel_size, time_size, ent_tuple, triples_idx_r, triples_idx_t, ent_ent, ent_ent_val, rel_ent, ent_rel, time_ent, ent_time= load_graph(path)
print("node_size & relation_size & time_size: %d, %d, %d." %(node_size,rel_size,time_size))

train_pair,test_pair = load_aligned_pair(path,ratio=seed_number)
candidates_x,candidates_y = set([x for x,y in test_pair]), set([y for x,y in test_pair]) 


# main functions of LightTEA

def random_projection(x,out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1],out_dim)),axis=-1)
    return K.dot(x,random_vec)

def batch_sparse_matmul(sparse_tensor,dense_tensor,batch_size = 128,save_mem = False):
    results = []
    for i in range(dense_tensor.shape[-1]//batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor,dense_tensor[:, i*batch_size:(i+1)*batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results,-1)
    else:
        return K.concatenate(results,-1)

def get_features_e(train_pair,extra_feature = None):
    
    if extra_feature is not None:
        ent_feature = extra_feature
    else:
        random_vec = K.l2_normalize(tf.random.normal((len(train_pair),ent_dim)),axis=-1)
        ent_feature = tf.tensor_scatter_nd_update(tf.zeros((node_size,ent_dim)),train_pair.reshape((-1,1)),tf.repeat(random_vec,2,axis=0))
    
    ent_ent_graph = tf.SparseTensor(indices=ent_ent,values=ent_ent_val,dense_shape=(node_size,node_size))

    ent_list = [ent_feature]
    for i in range(2):
        
        new_ent_feature = batch_sparse_matmul(ent_ent_graph,ent_feature)
        new_ent_feature = tf.nn.l2_normalize(new_ent_feature,axis=-1)        
        ent_feature = new_ent_feature
        ent_list.append(ent_feature)
    
    ent_feature = K.l2_normalize(K.concatenate(ent_list,1),-1).numpy()

    features = ent_feature
    
    faiss.normalize_L2(features)
    if extra_feature is not None:
        features = np.concatenate([ent_feature,features],axis=-1)
    return features

def get_features_r(train_pair,extra_feature = None):
    
    if extra_feature is not None:
        ent_feature = extra_feature
    else:
        random_vec = K.l2_normalize(tf.random.normal((len(train_pair),ent_dim)),axis=-1)
        ent_feature = tf.tensor_scatter_nd_update(tf.zeros((node_size,ent_dim)),train_pair.reshape((-1,1)),tf.repeat(random_vec,2,axis=0))
    rel_feature = tf.zeros((rel_size,ent_feature.shape[-1]))
    
    ent_ent_graph = tf.SparseTensor(indices=ent_ent,values=ent_ent_val,dense_shape=(node_size,node_size))
    rel_ent_graph = tf.SparseTensor(indices=rel_ent,values=K.ones(rel_ent.shape[0]),dense_shape=(rel_size,node_size))
    ent_rel_graph = tf.SparseTensor(indices=ent_rel,values=K.ones(ent_rel.shape[0]),dense_shape=(node_size,rel_size))
    
    ent_list,rel_list = [ent_feature],[rel_feature]
    for i in range(2):
        new_rel_feature = batch_sparse_matmul(rel_ent_graph,ent_feature)
        new_rel_feature = tf.nn.l2_normalize(new_rel_feature,axis=-1)
        
        new_ent_feature = batch_sparse_matmul(ent_ent_graph,ent_feature)
        new_ent_feature += batch_sparse_matmul(ent_rel_graph,rel_feature)
        new_ent_feature = tf.nn.l2_normalize(new_ent_feature,axis=-1)
        
        ent_feature = new_ent_feature; rel_feature = new_rel_feature
        ent_list.append(ent_feature); rel_list.append(rel_feature)
    
    ent_feature = K.l2_normalize(K.concatenate(ent_list,1),-1)
    rel_feature = K.l2_normalize(K.concatenate(rel_list,1),-1)
    rel_feature = random_projection(rel_feature,rel_dim)
    
    
    batch_size = ent_feature.shape[-1]//mini_dim
    sparse_graph = tf.SparseTensor(indices=triples_idx_r,values=K.ones(triples_idx_r.shape[0]),dense_shape=(np.max(triples_idx_r)+1,rel_size))
    adj_value = batch_sparse_matmul(sparse_graph,rel_feature)
    
    features_list = []
    for batch in range(rel_dim//batch_size + 1):
        temp_list = []
        for head in range(batch_size):
            if batch*batch_size+head>=rel_dim:
                break
            sparse_graph = tf.SparseTensor(indices=ent_tuple,values=adj_value[:,batch*batch_size+head],dense_shape=(node_size,node_size))
            feature = batch_sparse_matmul(sparse_graph,random_projection(ent_feature,mini_dim))
            temp_list.append(feature)
        if len(temp_list):
            features_list.append(K.concatenate(temp_list,-1).numpy())#[19054*1536,19054*1536,19054*1024]
    features = np.concatenate(features_list,axis=-1)
    
    faiss.normalize_L2(features)
    if extra_feature is not None:
        features = np.concatenate([ent_feature,features],axis=-1)
    return features

def get_features_t(train_pair,extra_feature = None):
    
    if extra_feature is not None:
        ent_feature = extra_feature
    else:
        random_vec = K.l2_normalize(tf.random.normal((len(train_pair),ent_dim)),axis=-1)
        ent_feature = tf.tensor_scatter_nd_update(tf.zeros((node_size,ent_dim)),train_pair.reshape((-1,1)),tf.repeat(random_vec,2,axis=0))
    time_feature = tf.zeros((time_size,ent_feature.shape[-1]))
    
    ent_ent_graph = tf.SparseTensor(indices=ent_ent,values=ent_ent_val,dense_shape=(node_size,node_size))
    time_ent_graph = tf.SparseTensor(indices=time_ent,values=K.ones(time_ent.shape[0]),dense_shape=(time_size,node_size))
    ent_time_graph = tf.SparseTensor(indices=ent_time,values=K.ones(ent_time.shape[0]),dense_shape=(node_size,time_size))
    
    ent_list,time_list = [ent_feature],[time_feature]
    for i in range(2):
        new_time_feature = batch_sparse_matmul(time_ent_graph,ent_feature)
        new_time_feature = tf.nn.l2_normalize(new_time_feature,axis=-1)
        
        new_ent_feature = batch_sparse_matmul(ent_ent_graph,ent_feature)
        new_ent_feature += batch_sparse_matmul(ent_time_graph,time_feature)
        new_ent_feature = tf.nn.l2_normalize(new_ent_feature,axis=-1)
        
        ent_feature = new_ent_feature; time_feature = new_time_feature
        ent_list.append(ent_feature); time_list.append(time_feature)
    
    ent_feature = K.l2_normalize(K.concatenate(ent_list,1),-1)
    time_feature = K.l2_normalize(K.concatenate(time_list,1),-1)
    time_feature = random_projection(time_feature,time_dim)
    
    
    batch_size = ent_feature.shape[-1]//mini_dim
    sparse_graph = tf.SparseTensor(indices=triples_idx_t,values=K.ones(triples_idx_t.shape[0]),dense_shape=(np.max(triples_idx_t)+1,time_size))
    adj_value = batch_sparse_matmul(sparse_graph,time_feature)
    
    features_list = []
    for batch in range(time_dim//batch_size + 1):
        temp_list = []
        for head in range(batch_size):
            if batch*batch_size+head>=time_dim:
                break
            sparse_graph = tf.SparseTensor(indices=ent_tuple,values=adj_value[:,batch*batch_size+head],dense_shape=(node_size,node_size))
            feature = batch_sparse_matmul(sparse_graph,random_projection(ent_feature,mini_dim))
            temp_list.append(feature)
        if len(temp_list):
            features_list.append(K.concatenate(temp_list,-1).numpy())
    features = np.concatenate(features_list,axis=-1)
    
    faiss.normalize_L2(features)
    if extra_feature is not None:
        features = np.concatenate([ent_feature,features],axis=-1)
    return features

def get_features(train_pair,alpha,extra_feature = None):
    rel_features = get_features_r(train_pair,extra_feature)
    time_features = get_features_t(train_pair,extra_feature)
    features = (1-alpha)*rel_features + alpha*time_features
    return features
 
# Obtain the structural features and iteratively generate Semi-supervised data
# "epoch = 1" represents removing the iterative strategy

epochs = 3
results = []
for epoch in range(epochs):
    print("Round %d start:"%(epoch+1))
    features = get_features(train_pair,alpha)  

    right_list,wrong_list,result = sparse_sinkhorn_test(test_pair,features,simt,beta,top_k)

    main_cost = time.time()-main_st
    print("Round %d cost time: %ds"%(epoch+1,main_cost))
    results.extend(result)
    results.extend([round(main_cost)])

    if epoch < epochs-1:
        left,right = list(candidates_x),list(candidates_y)
        if(len(left)<top_k):
                top_k = len(left)
        if(len(left)<test_pair.shape[0]):
            simt_candidates = get_simt_candidates(simt, test_pair, left, right)
            sims,index = sparse_sims(left,right,features,simt_candidates,beta,top_k,mode="test")
        else:    
            sims,index = sparse_sims(left,right,features,simt,beta,top_k,mode="test")
       
        sims,index = sinkhorn(sims,index)

        ranks = tf.argsort(-sims,-1).numpy()
        temp_pair = []
        x_list,y_list= list(candidates_x),list(candidates_y)
        for i in range(ranks.shape[0]):
            if sims[i,ranks[i,0]] > 0.8:
                x = x_list[i]
                y = y_list[index[i,ranks[i,0]]]
                temp_pair.append((x,y))

        for x,y in temp_pair:
            if x in candidates_x:
                candidates_x.remove(x)
            if y in candidates_y:
                candidates_y.remove(y)
        
        print("new generated pairs = %d"%(len(temp_pair)))
        print("rest pairs = %d\n"%(len(candidates_x)))
        
        if not len(temp_pair):
            break
        train_pair = np.concatenate([train_pair,np.array(temp_pair)])
    

