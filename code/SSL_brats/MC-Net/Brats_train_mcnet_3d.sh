#python ./code/Brats_train_mcnet_3d.py --dataset_name Brats19 --exp MCNet+ --model mcnet3d_v2 --max_iterations 30000 --labelnum 50 --gpu 1 --temperature 0.1
#python ./code/Brats_train_mcnet_3d.py --dataset_name Brats19 --exp MCNet+ --model mcnet3d_v2 --max_iterations 30000 --labelnum 25 --gpu 2 --temperature 0.1
## 앞으로 돌려야하는 것
python ./code/Brats_train_mcnet_3d_kfold.py --dataset_name Brats19_kfold --exp MCNet+_kfold --model mcnet3d_v2 --max_iterations 30000 --labelnum 56 --gpu 4 --temperature 0.1 --fold 5