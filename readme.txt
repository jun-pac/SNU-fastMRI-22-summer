
## Model description.
As a result of testing about 10 models including the model we implemented using WGAN, we found that the highest score was recorded when E2Evarnet[1] and data augmentation[2] were used together.
We've extensively explored the various settings of varnet, and even modified the behavior of varnet to better adapt to a given dataset. This is the reason for separating the fastMRI and fastmri modules.

The locations of codes related to varnet are as follows:
/root/fastMRI/utils/pl_modules/varnet_module.py

The locations of codes related to data augmentation are as follows:
/root/fastMRI/utils/mraugment/data_augment.py
/root/fastMRI/utils/mraugment/helpers.py
/root/fastMRI/utils/data/DA_transforms.py
/root/fastMRI/utils/pl_modules/data_module.py



## Preprocessing (Add image label and image mask to kspace file.)
Enter following command:
python fastMRI/Code/data_preparation.py



## Training
As the epoch increases, you should train with two different args settings.
The two settings are:

1. Epoch 0~30
python fastMRI/Code/train.py --aug_delay 4 --aug_strength 0.5 --aug_max_rotation 180 --aug_max_shearing-x 15.0 --aug_max_shearing-y 15.0 

When you reach epoch 30, press ^c to pause training.

2. Epoch 31~
python fastMRI/Code/train.py --aug_delay 17 --aug_strength 1.0 --aug_max_rotation 10 --aug_max_shearing-x 5.0 --aug_max_shearing-y 5.0



## Evaluating
Enter following command(This will save the results to /root/output/reconstructions): 
python fastMRI/Code/evaluate.py --state_dict_file /root/fastMRI/model/epoch=34.ckpt --data_path /root/input/leaderboard/kspace

And enter the following command:
python fastMRI/Code/leaderboard_eval.py 



## References
[1] Sriram, Anuroop, et al. "End-to-end variational networks for accelerated MRI reconstruction." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020. (https://github.com/facebookresearch/fastMRI)
[2] Fabian, Zalan, Reinhard Heckel, and Mahdi Soltanolkotabi. "Data augmentation for deep learning based accelerated MRI reconstruction with limited data." International Conference on Machine Learning. PMLR, 2021. (https://github.com/MathFLDS/MRAugment)
