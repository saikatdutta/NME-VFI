# Non-linear Motion Estimation for Video Frame Interpolation using Space-time Convolutions

[Arxiv](https://arxiv.org/abs/2201.11407)

**Repo under construction!**

## Requirements

- torch==1.1.0 (CUDA 10.1)
- torchvision==0.3.0
- opencv-python==3.4.2
- scikit-image==0.17.2


## Setup

Please setup [IRR](https://github.com/visinf/irr) repository and update installation directory in `model.py`.

## Datasets

- [Vimeo Septuplet](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)
- [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)
- [HD](https://sites.google.com/view/wenbobao/memc-net?authuser=0) 
- [GoPro](https://drive.google.com/file/d/1rJTmM9_mLCNzBUUhYIGldBYgup279E_f/view)

The quintuplets used for evaluation are stored in `datasets` folder as `.csv` files. Please modify the absolute path accordingly.

## Inference 

```
python eval.py --dataset <dataset name> --data_root <dataset location>
```


## References

Our code is built upon the following existing papers and repositories.

- [QVI](https://sites.google.com/view/xiangyuxu/qvi_nips19) 

- [IRR](https://github.com/visinf/irr) 

- [FLAVR](https://tarun005.github.io/FLAVR/)

- [SuperSloMo](https://github.com/avinashpaliwal/Super-SloMo)


### Contact:

`<github username>`779@gmail.com