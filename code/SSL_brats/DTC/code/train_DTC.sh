#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_1 --max_iterations=20000 --num_classes=3 &&
#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_2 --max_iterations=20000 --num_classes=3
#python Brats2_MT_UA2avg_2task_SDM_wo2Dec.py --exp='MT_UA2avg_2task_SDM_lateB' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2
#python Brats2_MT.py --exp='MT' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2
#python LA_train_dtc.py --exp='DTC' --model=vnet_3D_112_112_80_18_4 --max_iterations=30000 --gpu=5
python Brats_train_dtc_original_kfold.py --exp='DTC_kfold' --model=vnet_3D_96_32 --max_iterations=30000 --labelnum=56 --total_data_num=268 --gpu=4 --fold=4