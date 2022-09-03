# SNU-fastMRI-22-summer
**UPD : We placed 5th in the SNU fastMRI challenge 2022!** <br/>
(http://fastmri.snu.ac.kr/)

Final SSIM score : 0.9858 (Epoch 34)



## Model description.
As a result of testing about 10 models including the model we implemented using WGAN, we found that the highest score was recorded when E2Evarnet[1] and data augmentation[2] were used together.<br/>
We've extensively explored the various settings of varnet, and even modified the behavior of varnet to better adapt to a given dataset. This is the reason for separating the fastMRI and fastmri modules.

The locations of codes related to varnet are as follows:<br/>
>/root/fastMRI/utils/pl_modules/varnet_module.py

The locations of codes related to data augmentation are as follows:<br/>
>/root/fastMRI/utils/mraugment/data_augment.py<br/>
>/root/fastMRI/utils/mraugment/helpers.py<br/>
>/root/fastMRI/utils/data/DA_transforms.py<br/>
>/root/fastMRI/utils/pl_modules/data_module.py

---

## Preprocessing (Add image label and image mask to kspace file.)
Enter following command:<br/>
```
python fastMRI/Code/data_preparation.py
```


## Training
As the epoch increases, you should train with two different args settings. <br/>
The two settings are:

1. Epoch 0~30<br/>
```
python fastMRI/Code/train.py --aug_delay 4 --aug_strength 0.5 --aug_max_rotation 180 --aug_max_shearing-x 15.0 --aug_max_shearing-y 15.0 
```
When you reach epoch 30, press ^c to pause training.

2. Epoch 31~40<br/>
```
python fastMRI/Code/train.py --aug_delay 17 --aug_strength 1.0 --aug_max_rotation 10 --aug_max_shearing-x 5.0 --aug_max_shearing-y 5.0
```


## Evaluating<br/>
Enter following command(This will save the results to /root/output/reconstructions):<br/>
```
python fastMRI/Code/evaluate.py --state_dict_file /root/fastMRI/model/epoch=34.ckpt --data_path /root/input/leaderboard/kspace
```

And enter the following command:<br/>
```
python fastMRI/Code/leaderboard_eval.py 
```

---

## Note
One downside is that the data augmentation started too late and the training did not progress to the end. After various attempts, data augmentation was applied three days before the deadline. The training continued without interruption, but the competition ended before reaching the convergence value.<br/>
For personal regret and curiosity, we continued to train with our computer after the competition was over. I will not comment on the results obtained after the deadline as this may undermine the fairness of the competition. But in the sense of sharing the research results, I think it can be said that "we got very good results".

---

## References
[1] Sriram, Anuroop, et al. "End-to-end variational networks for accelerated MRI reconstruction." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020. (https://github.com/facebookresearch/fastMRI) <br/>
[2] Fabian, Zalan, Reinhard Heckel, and Mahdi Soltanolkotabi. "Data augmentation for deep learning based accelerated MRI reconstruction with limited data." International Conference on Machine Learning. PMLR, 2021. (https://github.com/MathFLDS/MRAugment)
