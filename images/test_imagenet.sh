# python run.py --arch resnet18 --avg_model True --data imagenet --data_dir /mlodata1/tlin/dataset/ILSVRC/raw_data/ --eval_freq 1 --num_epochs 90 --learning_rate 0.1 --lr_decay_ratios 0.33,0.33,0.33 --weight_decay 1e-4 --local_step 8 --block_step 1 --batch_size 128 --num_workers 0 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --block_size 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 --device imagenet --hostfile kubernetes/hostfile --mpi_path /home/lin/.openmpi/ --python_path /home/lin/conda/envs/pytorch-py3.6/bin/python

# python run.py --arch resnet18 --avg_model True --data imagenet --data_dir /mlodata1/tlin/dataset/ILSVRC/raw_data/ --eval_freq 1 --num_epochs 90 --learning_rate 0.1 --weight_decay 1e-4 --local_step 8 --block_step 1 --batch_size 128 --num_workers 0 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --block_size 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 --device imagenet --hostfile kubernetes/hostfile --mpi_path /home/lin/.openmpi/ --python_path /home/lin/conda/envs/pytorch-py3.6/bin/python

# python run.py --arch resnet18 --avg_model True --data imagenet --data_dir /mlodata1/tlin/dataset/ILSVRC/raw_data/ --eval_freq 1 --num_epochs 90 --learning_rate 0.1 --lr_decay_factor 10 --weight_decay 1e-4 --local_step 8 --block_step 1 --batch_size 128 --num_workers 0 --world 0,0,0,0,0,0,0,0,0,0,0,0 --block_size 1,1,1,1,1,1,1,1,1,1,1,1 --device imagenet --hostfile kubernetes/hostfile --mpi_path /home/lin/.openmpi/ --python_path /home/lin/conda/envs/pytorch-py3.6/bin/python

# python run.py --arch resnet18 --avg_model True --data imagenet --data_dir /mlodata1/tlin/dataset/ILSVRC/raw_data/ --eval_freq 1 --num_epochs 90 --learning_rate 1.2 --lr_warmup True --weight_decay 1e-4 --local_step 1 --block_step 1 --batch_size 128 --num_workers 0 --world 0,0,0,0,0,0,0,0,0,0,0,0 --block_size 1,1,1,1,1,1,1,1,1,1,1,1 --device imagenet --hostfile kubernetes/hostfile --mpi_path /home/lin/.openmpi/ --python_path /home/lin/conda/envs/pytorch-py3.6/bin/python

python run.py --arch resnet18 --avg_model True --data imagenet --data_dir /mlodata1/tlin/dataset/ILSVRC/raw_data/ --eval_freq 1 --num_epochs 90 --learning_rate 1.2 --lr_decay_ratios 0.33,0.33,0.33 --in_momentum False --out_momentum True --weight_decay 1e-4 --local_step 1 --block_step 1 --batch_size 128 --num_workers 0 --world 0,0,0,0,0,0,0,0,0,0,0,0 --block_size 1,1,1,1,1,1,1,1,1,1,1,1 --device imagenet --hostfile kubernetes/hostfile --mpi_path /home/lin/.openmpi/ --python_path /home/lin/conda/envs/pytorch-py3.6/bin/python



# mini-batch SGD with batch size 128.
# without data shuffle, set lr=0.1 and with lr decay. failed. 60%
# with data shuffle, set lr=0.1 and with lr decay. failed, 60%
# linearly scaled learning rate from 0.1 to 1.2 without warmup, failed.
# linearly scaled learning rate from 0.1 to 1.2 with warmup, failed.
# turn off in momentum and use out momentum instead. use constant lr=1.2, without scaling, failed and stopped.
# turn off in momentum and use out momentum instead. use constant lr=1.2 with scaling.
