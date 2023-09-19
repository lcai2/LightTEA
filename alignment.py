import numpy as np
import tensorflow as tf
import faiss
import tensorflow.keras.backend as K
from args import args
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


def get_simt_candidates(simt,test_pair,left,right):
    dict_left = {}
    for i,v in enumerate(test_pair[:,0]):
        dict_left[v] = i
    
    dict_right = {}
    for i,v in enumerate(test_pair[:,1]):
        dict_right[v] = i

    left_index = []
    for v in left:
        left_index.append(dict_left[v])

    right_index = []
    for v in right:
        right_index.append(dict_right[v])

    batch_size = 512
    simt_list = []
    for i in range(simt.shape[0]//batch_size + 1):        
        simt_temp = tf.gather(simt[i*batch_size:(i+1)*batch_size],right_index,axis=1).numpy()
        simt_list.append(simt_temp)
    simt = np.concatenate(simt_list)
    simt = tf.gather(simt,left_index,axis=0)
    return simt

def sparse_sims(left,right,features,simt,beta,top_k=500,mode = "test"):
    features_l = features[left]
    features_r = features[right]

    faiss.normalize_L2(features_l); 
    faiss.normalize_L2(features_r)

    res = faiss.StandardGpuResources()
    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    if mode == "test":
        param = 'Flat'
        index = faiss.index_factory(dim, param, measure)
    else:
        param = 'IVF256(RCQ2x5),PQ32'
        index = faiss.index_factory(dim, param, measure)
        index.nprobe = 16
    if len(gpus):
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(features_r)
    index.add(features_r)
    sims,index= index.search(features_l, top_k)
    if simt is not None:
        batch_size = 512
        simt_list = []
        for i in range(simt.shape[0]//batch_size + 1):        
            simt_temp = tf.gather(simt[i*batch_size:(i+1)*batch_size],index[i*batch_size:(i+1)*batch_size],axis=1,batch_dims=1).numpy()
            simt_list.append(simt_temp)
        simt = np.concatenate(simt_list)

        sims = (1-beta)*sims + beta*simt
    return sims,index

def sinkhorn(sims,index,iteration=15):
    row_sims = K.exp(sims.flatten()/0.02)
    index = K.flatten(index.astype("int32"))

    size = sims.shape[0]
    row_index = K.transpose(([K.arange(size*sims.shape[1])//sims.shape[1],index,K.arange(size*sims.shape[1])]))
    col_index = tf.gather(row_index,tf.argsort(row_index[:,1]))
    covert_idx = tf.argsort(col_index[:,2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:,0],params = tf.math.segment_sum(row_sims,row_index[:,0]))
        col_sims = tf.gather(row_sims,col_index[:,2])
        col_sims = col_sims / tf.gather(indices=col_index[:,1],params = tf.math.segment_sum(col_sims,col_index[:,1]))
        row_sims = tf.gather(col_sims,covert_idx)
        
    index = K.reshape(row_index[:,1],(-1,sims.shape[1])).numpy()
    sims = K.reshape(row_sims,(-1,sims.shape[1])).numpy()
    
    return sims,index

def sparse_sinkhorn_test(test_pair,features,simt=None,beta=0.3,top_k=500,iteration=15):
    left, right = test_pair[:,0], test_pair[:,1]
    sims,index = sparse_sims(left,right,features,simt,beta,top_k,mode="test")
    sims,index = sinkhorn(sims,index,iteration)
    ranks = tf.argsort(-sims,-1).numpy()
    
    wrong_list,right_list = [],[]
    h1,h10,mrr = 0, 0, 0
    pos = np.zeros(np.max(right)+1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i,1]] == index[i,ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i],right[index[i,ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1/(rank[0]+1) 
    print("MRR: %.3f Hits@1: %.3f Hits@10: %.3f"%(mrr/len(test_pair),h1/len(test_pair),h10/len(test_pair)))
    result = [round(mrr/len(test_pair),3),round(h1/len(test_pair),3),round(h10/len(test_pair),3)]
    return right_list, wrong_list, result

