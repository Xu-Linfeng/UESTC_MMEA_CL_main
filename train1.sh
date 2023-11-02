################ Flow

# python main.py mydataset Flow --config ./exps/myfinetune.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth --flow_prefix flow_ 

# python main.py mydataset Flow --config ./exps/mylwf.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth --flow_prefix flow_ 

# python main.py mydataset Flow --config ./exps/myicarl.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --arch BNInception --num_segments 8 --dropout 0.5 --epochs 50 -b 8 \
# --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn  -j 8 \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth --flow_prefix flow_ 


############### Flow + Gyro
# python main.py mydataset Flow Gyro --config ./exps/myfinetune.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --mpu_path '/data_25T/whx/temporal-binding-network/dataset/gyro/' --arch BNInception --num_segments 8 \
# --dropout 0.5 --epochs 50 -b 8 --lr 0.001 --lr_steps 10 20 --flow_prefix flow_ --gd 20 --partialbn \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth -j 8

# python main.py mydataset Flow Gyro --config ./exps/mylwf.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --mpu_path '/data_25T/whx/temporal-binding-network/dataset/gyro/' --arch BNInception --num_segments 8 \
# --dropout 0.5 --epochs 50 -b 8 --lr 0.001 --lr_steps 10 20 --flow_prefix flow_ --gd 20 --partialbn \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth -j 8

# python main.py mydataset Flow Gyro --config ./exps/myicarl.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --mpu_path '/data_25T/whx/temporal-binding-network/dataset/gyro/' --arch BNInception --num_segments 8 \
# --dropout 0.5 --epochs 50 -b 8 --lr 0.001 --lr_steps 10 20 --flow_prefix flow_ --gd 20 --partialbn \
# --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth -j 8


################# RGB + Gyro

# python main.py mydataset RGB Gyro --config ./exps/myicarl.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --mpu_path '/data_25T/whx/temporal-binding-network/dataset/gyro/' --arch BNInception --num_segments 8 \
# --dropout 0.5 --epochs 50 -b 8 --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn -j 8

# python main.py mydataset RGB Gyro --config ./exps/myfinetune.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --mpu_path '/data_25T/whx/temporal-binding-network/dataset/gyro/' --arch BNInception --num_segments 8 \
# --dropout 0.5 --epochs 50 -b 8 --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn -j 8

# python main.py mydataset RGB Gyro --config ./exps/mylwf.json \
# --train_list mydataset_train.txt --val_list mydataset_test.txt \
# --mpu_path '/data_25T/whx/temporal-binding-network/dataset/gyro/' --arch BNInception --num_segments 8 \
# --dropout 0.5 --epochs 50 -b 8 --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn -j 8

python main.py mydataset RGB Gyro Acce  --config ./exps/myewc.json --train_list mydataset_train.txt --val_list mydataset_test.txt --mpu_path '/home/amax/Downloads/whx/temporal-binding-network/dataset/gyro/' --arch BNInception --num_segments 8 --dropout 0.5 --epochs 30 -b 8 --lr 0.001 --lr_steps 10 20 --gd 20 --partialbn -j 8 --pretrained_flow ./pretrain_model/TSN-kinetics-flow.pth