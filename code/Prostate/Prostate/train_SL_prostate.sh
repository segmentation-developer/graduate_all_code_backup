#python HV_train_SL.py --exp='HV_result/Unet2_SL_labelCrop' --denseUnet_3D=unet_newdata_20000_1 --max_iterations=20000 --num_classes=3
#python pancreas_train_SL.py --exp='pancreas_result/Unet2_SL_imageCrop' --denseUnet_3D=unet_newdata_20000_1 --max_iterations=20000 --num_classes=3
#python train.py --exp='prostate_result/MSD_2D_SL/SL_bs48' --model=TansUnet_10000_1 --max_iterations=10000 --labeled_bs=2 --num_classes=3 --sw_batch_size=8 --overlap=0.5
#python SSL_Prostate_train_1class_GDT_MT.py --exp='SSL/GDT-MT_350_350_200_rampup_refpaper' --model=Vnet_3D_208_randomCrop --batch_size=2 --max_iterations=30000 --num_classes=2 --class_name=1
#python Prostate_train_SL_1class_woSlidingW.py --exp='SL/sameForm_Wssl'  --model=Vnet_3D_256_randomCrop --batch_size=2 --max_iterations=30000 --num_classes=2 --class_name=1 --fold=1
python SSL_Prostate_train_1class_kfold.py --exp='SSL/MT_ATO_350_350_200_rampup_refpaper' --model=Vnet_3D_256_randomCrop --batch_size=2 --max_iterations=30000 --num_classes=2 --class_name=1 --fold=3
