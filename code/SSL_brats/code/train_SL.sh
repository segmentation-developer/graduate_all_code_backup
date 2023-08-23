python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT_opti' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --num_classes=2 --gpu=5 --fold=1 &&
python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT_opti' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=28 --consistency=1.0 --num_classes=2 --gpu=5 --T=6 --fold=5 &&
python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT_opti' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --num_classes=2 --gpu=5 --fold=3 &&
python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT_opti' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --num_classes=2 --gpu=5 --fold=4 &&
python Brats2_GDT_MT_kfold.py --exp='kfold/GDT-MT_opti' --model=vnet_3D_96_32 --max_iterations=30000 --labeled_num=56 --num_classes=2 --gpu=5 --fold=5
