echo "Start Train"
dir=$(dirname "$0")
if [ -f "$dir/Train.py" ];then
    cd $dir
    pwd
    export CUDA_VISIBLE_DEVICES=0
    ##### run with Polyp Datasets
    python Train.py --model_name CSCAUNet --epoch 121 --batchsize 16 --trainsize 352 --train_save CSCAUNet_Kvasir_1e4_bs16_e120_s352 --lr 0.0001 --train_path $dir/data/TrainDataset --test_path $dir/data/TestDataset/Kvasir/   
    sleep 1m
    python Test.py --train_save CSCAUNet_Kvasir_1e4_bs16_e120_s352 --testsize 352 --test_path $dir/data/TestDataset
    
    ### run with ISIC 2018
    python Train.py --model_name CSCAUNet --epoch 121 --batchsize 16 --trainsize 352 --train_save CSCAUNet_ISIC_1e4_bs16_e120_s352 --lr 0.0001 --train_path $dir/data/ISIC2018/train --test_path $dir/data/ISIC2018/val/   
    sleep 1m
    python Test.py --train_save CSCAUNet_ISIC_1e4_bs16_e120_s352 --testsize 352 --test_path $dir/data/ISIC2018

    ### run with 2018 DSB
    python Train.py --model_name CSCAUNet --epoch 121 --batchsize 16 --trainsize 256 --train_save CSCAUNet_DSB_1e4_bs16_e120_s256 --lr 0.0001 --train_path $dir/data/2018DSB/train --test_path $dir/data/2018DSB/val/   
    sleep 1m
    python Test.py --train_save CSCAUNet_ISIC_1e4_bs16_e120_s256 --testsize 256 --test_path $dir/data/2018DSB
else
    echo "file not exists"
fi
