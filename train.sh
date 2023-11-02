################# RGB

# python main.py mydataset RGB --config ./exps/myfinetune.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8

# python main.py mydataset RGB --config ./exps/mylwf.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8

# python main.py mydataset RGB --config ./exps/myewc.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8

# python main.py mydataset RGB --config ./exps/myicarl.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8

############### Gyro

# python main.py mydataset Gyro --config ./exps/myfinetune.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 32 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
# --mpu_path '/home/amax/Downloads/whx/temporal-binding-network/dataset/gyro/'

#python main.py mydataset Acce --config ./exps/myicarl.json \
#--train_list mydataset_train.txt --val_list mydataset_test.txt \
#--arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 32 \
#--lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
#--mpu_path '/home/amax/Downloads/whx/temporal-binding-network/dataset/gyro/'

# python main.py mydataset Gyro --config ./exps/myicarl.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 32 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
# --mpu_path '/data_25T/whx/temporal-binding-network/dataset/gyro/'


############### RGB + Flow

# python main.py mydataset RGB Flow --config ./exps/myfinetune.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth --flow_prefix flow_ 

# python main.py mydataset RGB Flow --config ./exps/mylwf.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth --flow_prefix flow_ 



