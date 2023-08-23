#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_1 --max_iterations=20000 --num_classes=3 &&
#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_2 --max_iterations=20000 --num_classes=3
#python Brats2_MT_UA2avg_2task_SDM_wo2Dec.py --exp='MT_UA2avg_2task_SDM_lateB' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2
#python Brats2_MT.py --exp='MT' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2
python Brats2_MT.py --exp='MT' --model=ssl4mis_vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2
