import numpy as np
import os
from multiprocessing import Pool
from utils import *

def get_time_dict(file_path):
    entity,rel,time,quadruples=load_quadruples(file_path)
    time_dict = {}    
    for i in range(max(entity)+1):
        time_dict[i] = []
    for head,r,tail,t in quadruples:
        if(t!=0):
            time_dict[head].append(t); time_dict[tail].append(t)
    return time_dict

thread_num=30 

def list2dict(time_list):
    dic={}
    for i in time_list:
        dic[i]=time_list.count(i)
    return dic

def sim_matrix(t1,t2):
    size_t1 = len(t1)
    size_t2 = len(t2)
    matrix = np.zeros((size_t1,size_t2))
    for i in range(size_t1):
        len_a = sum(t1[i].values())
        for j in range(size_t2):
            len_b = sum(t2[j].values())
            len_ab = len_a + len_b
            set_ab = {}
            set_ab = t1[i].keys() & t2[j].keys()
            if (len(set_ab)==0):
                matrix[i,j] = 0
                continue
            count = 0
            for k in set_ab:
                count = count + (min(t1[i][k],t2[j][k])-1)
            count = len(set_ab) + count
            matrix[i,j] = (count*2) / len_ab
    return matrix

def div_array(arr,n):
    arr_len = len(arr)
    k = arr_len // n
    ls_return = []
    for i in range(n-1):
        ls_return.append(arr[i*k:i*k+k])
    ls_return.append(arr[(n-1)*k:])
    return ls_return

def thread_sim_matrix(t1,t2):
    pool = Pool(processes=thread_num)
    reses = list()
    tasks_t1 = div_array(t1,thread_num)    
    for task_t1 in tasks_t1:
        reses.append(pool.apply_async(sim_matrix,args=(task_t1,t2)))
    pool.close()
    pool.join()
    matrix = None
    for res in reses:
        val = res.get()
        if matrix is None:
            matrix = np.array(val)
        else:
            matrix = np.concatenate((matrix,val),axis=0)
    return matrix
    
def pair_simt(time_dict,pair):
    t1 = [list2dict(time_dict[e1]) for e1, e2 in pair]
    t2 = [list2dict(time_dict[e2]) for e1, e2 in pair]
    m = thread_sim_matrix(t1,t2)
    return m

def cal_simt(file_name,pair):    
    if os.path.exists(file_name):
        pair_mt = np.load(file_name)
    else:        
        time_dict = get_time_dict(os.path.dirname(os.path.dirname(file_name)) + "/")
        pair_mt = pair_simt(time_dict,pair)
        np.save(file_name,pair_mt)
    return pair_mt

datasets=["ICEWS05-15","YAGO-WIKI"] 
dataset = datasets[0] # select dataset
seed_num = 1000 # set the number of seeds
file_path = "./data/" + dataset + "/"
file_name = file_path + "simt/simt_" + dataset + "_" + str(seed_num) + ".npy"

tain_pair, test_pair = load_aligned_pair(file_path,ratio = seed_num)
print("generating...")
simt = cal_simt(file_name,test_pair)
print("done!")
