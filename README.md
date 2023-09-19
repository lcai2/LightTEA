# LightTEA
Codes for IJCAI2023 paper [“An Effective and Efficient Time-aware Entity Alignment Framework via Two-aspect Three-view Label Propagation”](https://www.ijcai.org/proceedings/2023/0558.pdf)

# Environment
python                    3.7.11   
tensorflow-gpu       2.6.0  
keras                       2.6.0  
cudatoolkit             11.3.1  
cudnn                     8.1.0.77   
faiss-gpu                1.7.2  

# Usage
On the first run,  you need to use command "python cal_simt.py" to calculate the temporal similarity matrix, then use command "python LightTEA.py" to get the results. The first run may be slow because the graph needs to be preprocessed into binary cache.

For future runs, you only need to use command "python LightTEA.py" to get the results.

# Acknowledgement
We refer to the code of LightEA. Thanks for their great contributions!

# Citation
If you use this model or code, please cite it as follows:  
@inproceedings{Cai23_lightTEA,  
  author       = {Li Cai and
                  Xin Mao and
                  Youshao Xiao and
                  Changxu Wu and
                  Man Lan},  
  title        = {An Effective and Efficient Time-aware Entity Alignment Framework via
                  Two-aspect Three-view Label Propagation},  
  booktitle    = {Proceedings of the Thirty-Second International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2023, 19th-25th August 2023, Macao,
                  SAR, China},  
  pages        = {5021--5029},  
  publisher    = {ijcai.org},  
  year         = {2023},  
  url          = {https://doi.org/10.24963/ijcai.2023/558},  
  doi          = {10.24963/ijcai.2023/558},  
}
