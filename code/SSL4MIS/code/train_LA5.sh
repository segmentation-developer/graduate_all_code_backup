#python Brats_train_Unet2_dsvURUM_4to1_boundary.py --exp='brats19_result/Unet2_dsv4URUM_boundary' --model=unet_newdata_30000_1 --max_iterations=30000 --num_classes=2 --gpu=5 &&
#python Brats_train_Unet2_dsvURUM_4to1_boundary.py --exp='brats19_result/Unet2_dsv4URUM_boundary' --model=unet_newdata_30000_2 --max_iterations=30000 --num_classes=2 --gpu=5
#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_1 --max_iterations=20000 --num_classes=3 &&
#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_2 --max_iterations=20000 --num_classes=3
python Brats2_UAMT_ConvNext.py --exp='UAMT' --model=vnet_3D_96 --max_iterations=30000 --num_classes=2