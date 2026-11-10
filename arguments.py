import argparse

parser = argparse.ArgumentParser(description='train the model')

# 训练设置参数
parser.add_argument('--resume', default=False, help='# resume the training')
parser.add_argument('--resume_file_name', default=1, help='# the file resume the training from')

parser.add_argument('--lr_m', default='cosine', help='# cosine or index')
parser.add_argument('--cropsize', type=tuple, default=(512, 384), help='# image crop size')

parser.add_argument('--vis_path', type=str, default='./datasets/MSRS/Visible', help='# Visible image path')
parser.add_argument('--inf_path', type=str, default='./datasets/MSRS/Infrared', help='# Infrared image path')
parser.add_argument('--lab_path', type=str, default='./datasets/MSRS/Label', help='# Label image path')
# 训练过程 超参数
# 融合训练
parser.add_argument('--F_epoch', type=int, default=15, help='# number of epoch in fusion training')  # 15
parser.add_argument('--F_batch_size', type=int, default=2, help='# batch size in fusion training')
parser.add_argument('--F_num_workers', type=int, default=2, help='# num_workers in fusion training')
parser.add_argument('--F_lr', type=float, default=3e-3, help='# initial learning rate in fusion training')
parser.add_argument('--F_lr_decay', type=float, default=0.9, help='# learning rate decay in fusion training')
# 前向融合
parser.add_argument('--F_batch_size_', type=int, default=4, help='# fuse forward inference')
parser.add_argument('--F_num_workers_', type=int, default=4, help='# fuse forward inference')
# 分割
parser.add_argument('--S_epoch', type=int, default=80, help='# total epoch in segmentation training')  # 80
parser.add_argument('--S_warmup_steps', type=int, default=1000, help='# steps of warm up  in segmentation training')  # 1000
parser.add_argument('--S_batch_size', type=int, default=8, help='# batch size in segmentation training')
parser.add_argument('--S_num_workers', type=int, default=2, help='# num_workers in segmentation training')
parser.add_argument('--S_lr', type=float, default=1e-2, help='# initial learning rate in segmentation training')
parser.add_argument('--S_warmup_start_lr', type=float, default=1e-5,
                    help='# warm up start learning rate in segmentation training')
parser.add_argument('--S_score_thres', type=float, default=0.7, help='# score threshold in segmentation training')

# # 损失参数
# parser.add_argument('--int_loss_type', default='max', help='# int loss type : max、avg、text、var')
# parser.add_argument('--loss_weight', type=tuple, default=(10, 20, 30), help='# loss of ssim、text、int')

args = parser.parse_args()
# print(args)
