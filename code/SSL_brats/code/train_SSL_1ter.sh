#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_1 --max_iterations=20000 --num_classes=3 &&
#python HV_train_Unet2_dsv1URUM_boundary.py --exp='HV_result/Unet2_dsv1ce' --model=unet_newdata_20000_2 --max_iterations=20000 --num_classes=3
#python Brats2_MT_UA2avg_2task_SDM_wo2Dec.py --exp='MT_UA2avg_2task_SDM_lateB' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2
#python Brats2_MT.py --exp='MT' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2
#python Brats2_MT_UA2avg_2task_SDM_2Dec_SLandSSL.py --exp='MT_UA2avg_2task_SDM_dualD_SLandSSL' --model=vnet_3D_96 --max_iterations=25000 --sl_max_iterations=5000 --labeled_num=25 --num_classes=2
#python Brats2_MT_UA2avg_2task_SDM_2Dec_allDataConsistencyLoss.py --exp='Consis1.0_MT_UA2avg_2task_SDM_dualD_ConsisAllData' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2 --gpu=2 --T=8
#python Brats2_MT_UA2avg_2task_SDM_2Dec.py --exp='Consis1.0_MT_UA2avg_2task_SDM_dualD' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2 --gpu=3 --T=8
#python Brats2_MT_UA2avg_2task_SDM_2Dec_Uncertainty_ConsisModi_sharpening.py --exp='MT_UA2avg_2task_SDM_dualD_sharpening' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2 --gpu=1 --T=8 --consistency=0.1
#python Brats2_MT_UA2avg_2task_SDM_2Dec_UncertaintyAndSharpening_ConsisModi.py --exp='Consis1.0_MT_UA2avg_2task_SDM_dualD_UncertaintyAndSharpening' --model=vnet_3D_96 --max_iterations=30000 --labeled_num=25 --num_classes=2 --gpu=0 --T=8 --consistency=1.0
#python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --consistency=0.1 --num_classes=2 --gpu=4 --T=8 --fold=5
#python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --consistency=1.0 --num_classes=2 --gpu=2 --T=8 --fold=5
####여기서부터 돌리면 됨
#python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=28 --consistency=1.0 --num_classes=2 --gpu=5 --T=6 --fold=5
#python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=28 --consistency=1.0 --num_classes=2 --gpu=4 --T=4 --fold=5
#python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=28 --consistency=0.3 --num_classes=2 --gpu=3 --T=6 --fold=5
#python Brats2_GDT-MT_woATO_kfold.py --exp='kfold/GDT-MT' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=28 --consistency=?? --num_classes=2 --gpu=? --T=1 --fold=5
#python Brats2_GDT-MT_woSDM_kfold.py --exp='kfold/GDT-MT_woSDM' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=28 --consistency=?? --num_classes=2 --gpu=? --T=? --fold=5
#### SDM validation -> network change

#python Brats2_GDT-MT_woSDM.py --exp='GDT-MT_woSDM' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=50 --consistency=0.1 --num_classes=2 --gpu=3 --T=6
#python Brats2_GDT-MT_woATO.py --exp='GDT-MT_woATO' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=50 --consistency=0.1 --num_classes=2 --gpu=4

python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT_opti' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --consistency=1.0 --num_classes=2 --gpu=2 --T=6 --fold=1 &&
python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT_opti' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --consistency=1.0 --num_classes=2 --gpu=2 --T=6 --fold=2
