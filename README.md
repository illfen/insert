# FreeInsert: Disentangled Text-Guided Object Insertion in 3D Gaussian Scene without Spatial Priors (ACMMM2025)

[Project page](https://tjulcx.github.io/FreeInsert/) |[Paper](https://arxiv.org/abs/2505.01322) 


## Dependencies
Install with pip:
```
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio===0.12.1+cu116
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    pip install diffusers==0.22.1
    pip install huggingface_hub==0.25.0
    pip install transformers==4.44.2
    pip install open3d==0.17.0 trimesh==3.22.5 pymeshlab
    
    # install gaussian rasterization
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
    pip install ./diff-gaussian-rasterization
    
    # install simple-knn
    git clone https://github.com/camenduru/simple-knn.git
    pip install ./simple-knn

```
```
    git clone https://github.com/facebookresearch/sam2.git florence-sam/sam2
    mv florence-sam/sam2/sam2/* florence-sam/sam2/
    rm -rf florence-sam/sam2/sam2
```


### Start training

1. Scene reconstruction & Parser Output
```
    bash add_sunglasses_to_the_doll_stage1.sh
```
2. Object Generation & DoF Initialization
```
    bash add_sunglasses_to_the_doll_stage2.sh
```
3. DoF refinement
```
    bash add_sunglasses_to_the_doll_stage3.sh
```
4. Appearance refinement
```
    bash add_sunglasses_to_the_doll_stage4.sh
```



## Citation
If you find this code helpful for your research, please cite:
```
@article{li2025freeinsert,
  title={FreeInsert: Disentangled Text-Guided Object Insertion in 3D Gaussian Scene without Spatial Priors},
  author={Li, Chenxi and Wang, Weijie and Li, Qiang and Lepri, Bruno and Sebe, Nicu and Nie, Weizhi},
  journal={arXiv preprint arXiv:2505.01322},
  year={2025}
}
```

## Acknowledgments
This code based on [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting),[TIP-Editor](https://github.com/zjy526223908/TIP-Editor), [Dreambooth](https://huggingface.co/docs/diffusers/training/dreambooth), [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/), [GaussianEditor](https://github.com/buaacyw/GaussianEditor). 

