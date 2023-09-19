import numpy as np
import os
import pickle

def load_quadruples(path):
    quadruples = []
    entity = set() 
    rel = set([0]) 
    time = set()
    with open(path + "triples_1") as f:
        for line in f.readlines():
            items = line.strip().split("\t")
            if len(items) == 4:
                head,r,tail,t = [int(item) for item in items] 
                entity.add(head); entity.add(tail); rel.add(2*r); rel.add(2*r+1); time.add(t)
                quadruples.append((head,tail,2*r,t)) 
                quadruples.append((tail,head,2*r+1,t)) 
            else:
                head,r,tail,tb,te = [int(item) for item in items]  
                entity.add(head); entity.add(tail); rel.add(2*r); rel.add(2*r+1); time.add(tb); time.add(te)  
                quadruples.append((head,tail,2*r,tb)) 
                quadruples.append((tail,head,2*r+1,tb)) 
                quadruples.append((head,tail,2*r,te)) 
                quadruples.append((tail,head,2*r+1,te)) 
    with open(path + "triples_2") as f:
        for line in f.readlines():
            items = line.strip().split("\t")
            if len(items) == 4:
                head,r,tail,t = [int(item) for item in items] 
                entity.add(head); entity.add(tail); rel.add(2*r); rel.add(2*r+1); time.add(t)
                quadruples.append((head,tail,2*r,t)) 
                quadruples.append((tail,head,2*r+1,t)) 
            else:
                head,r,tail,tb,te = [int(item) for item in items]  
                entity.add(head); entity.add(tail); rel.add(2*r); rel.add(2*r+1); time.add(tb); time.add(te)  
                quadruples.append((head,tail,2*r,tb)) 
                quadruples.append((tail,head,2*r+1,tb)) 
                quadruples.append((head,tail,2*r,te)) 
                quadruples.append((tail,head,2*r+1,te)) 
    return entity,rel,time,quadruples

def load_graph(path):
    
    if os.path.exists(path+"graph_cache.pkl"):
        return pickle.load(open(path+"graph_cache.pkl","rb"))
    
    entity, rel, time, quadruples = load_quadruples(path)
    quadruples = np.unique(quadruples,axis=0)
    node_size = max(entity) + 1  
    rel_size = max(rel) + 1   
    time_size = max(time) + 1
    
    
    ent_tuple,triples_idx_r,triples_idx_t= [],[],[] ###
    ent_ent_s,rel_ent_s,ent_rel_s = {},set(),set()    
    #
    time_ent_s, ent_time_s = set(), set() ###

    last,index = (-1,-1), -1

    for i in range(node_size):
        ent_ent_s[(i,i)] = 0  

    for head,tail,r,t in quadruples:
        ent_ent_s[(head,head)] += 1
        ent_ent_s[(tail,tail)] += 1

        if (head,tail) != last:
            last = (head,tail)
            index += 1
            ent_tuple.append([head,tail])
            ent_ent_s[(head,tail)] = 0

        triples_idx_r.append([index,r])
        triples_idx_t.append([index,t]) ###
        ent_ent_s[(head,tail)] += 1
        rel_ent_s.add((r,head))
        ent_rel_s.add((tail,r))
        #
        time_ent_s.add((t,head))
        ent_time_s.add((tail,t))

        
        

    ent_tuple = np.array(ent_tuple)
    triples_idx_r = np.unique(np.array(triples_idx_r),axis=0)
    # 
    triples_idx_t = np.unique(np.array(triples_idx_t),axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())),axis=0)
    ent_ent_val = np.array([ent_ent_s[(x,y)] for x,y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)),axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)),axis=0)
    ###
    time_ent = np.unique(np.array(list(time_ent_s)),axis=0)
    ent_time = np.unique(np.array(list(ent_time_s)),axis=0)
    ###
  
    graph_data = [node_size, rel_size, time_size, ent_tuple, triples_idx_r, triples_idx_t, ent_ent, ent_ent_val, rel_ent, ent_rel, time_ent, ent_time]#,rt_ent, ent_rt]
    pickle.dump(graph_data, open(path+"graph_cache.pkl","wb"))
    return graph_data

def load_aligned_pair(file_path,ratio = 1000):
    with open(file_path + "ref_pairs") as f:
        ref = f.readlines()
    try:
        with open(file_path + "sup_pairs") as f:
            sup = f.readlines()
    except:
        sup = None
    
    ref = np.array([line.replace("\n","").split("\t") for line in ref]).astype(np.int64)
    if sup:
        sup = np.array([line.replace("\n","").split("\t") for line in sup]).astype(np.int64)
        ref = np.concatenate([sup,ref])
    train_size = ratio
    return ref[:train_size],ref[train_size:]

