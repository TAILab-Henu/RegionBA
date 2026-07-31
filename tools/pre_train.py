import os
import sys
import torch
import numpy as np
import datetime
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
# 添加当前目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from data_utils.ModelNetDataLoader import (
    augment_point_cloud,
)
from data_utils.dataset_config import configure_dataset_args, create_clean_dataset
from model_config import (
    build_training_policy,
    get_clean_log_dir,
    import_model_module,
)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training clean model')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--model', default='dgcnn',
                        help='model name: dgcnn, pointnet++, or curvenet')
    parser.add_argument('--dataset', type=str, default='modelnet40',
                        help='dataset: modelnet10, modelnet40, or shapenetpart16')
    parser.add_argument('--num_category', default=None, type=int, choices=[10, 16, 40],
                        help='optional category-count validation; inferred from --dataset')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum used by DGCNN and CurveNet')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')

    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--data_root', type=str, default=None,
                        help='dataset root')
    parser.add_argument('--log_root', type=str, default='log',
                        help='root directory for clean-model experiments')
    return parser.parse_args()


def inplace_relu(m):
    """设置ReLU为原地操作"""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def test(model, loader, num_class=40, device='cuda'):
    """测试模型"""
    mean_correct = []
    model.eval()

    for data in tqdm(loader):
        points, target = data[0], data[1]

        if device == 'cuda':
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        try:
            pred, _ = model(points)
        except ValueError:
            pred = model(points)

        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    instance_acc = np.mean(mean_correct)
    return instance_acc


def main(args):
    configure_dataset_args(args)

    # 设置随机种子
    np.random.seed(256)
    torch.manual_seed(256)
    torch.cuda.manual_seed(256)

    def log_string(str):
        """日志记录"""
        logger.info(str)
        print(str)

    '''设置GPU'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu'

    '''创建目录'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = get_clean_log_dir(args.model, args.num_category, args.log_root)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)

    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''日志配置'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(args)

    '''数据加载'''
    log_string('Loading dataset ...')
    num_class = args.num_category

    log_string(f"Dataset: {args.dataset}")
    log_string(f"Data path: {args.data_root}")
    log_string(f"Using uniform sample: {args.use_uniform_sample}")

    try:
        train_dataset = create_clean_dataset(args, split='train', process_data=True)
        test_dataset = create_clean_dataset(args, split='test', process_data=True)
        log_string("Dataset loaded successfully")
    except Exception as e:
        log_string(f"Error loading dataset: {e}")
        raise

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)

    '''模型加载'''
    log_string('Loading model ...')
    model = import_model_module(args.model)
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if device == 'cuda':
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    '''优化器'''
    optimizer, scheduler, training_policy = build_training_policy(
        args.model,
        classifier.parameters(),
        learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        epochs=args.epoch,
        momentum=args.momentum,
    )
    log_string(
        f"Optimizer: {training_policy['optimizer']}, "
        f"effective lr: {training_policy['effective_learning_rate']}, "
        f"scheduler: {training_policy['scheduler']}"
    )
    augmentation_name = "random translation + point shuffle"
    log_string(f"Training augmentation: {augmentation_name}")

    '''训练'''
    log_string('Start training ...')
    best_acc = 0.0

    for epoch in range(args.epoch):
        log_string(f'Epoch {epoch + 1}/{args.epoch}:')
        mean_correct = []
        classifier = classifier.train()
        if not training_policy['scheduler_step_after_epoch']:
            scheduler.step()

        for _, (points, target, *_) in tqdm(enumerate(trainDataLoader, 0),
                                        total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = torch.from_numpy(augment_point_cloud(points.numpy()))
            points = points.transpose(2, 1)
            if device == 'cuda':
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            loss.backward()
            if training_policy['gradient_clip_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    classifier.parameters(),
                    training_policy['gradient_clip_norm'],
                )
            optimizer.step()

        if training_policy['scheduler_step_after_epoch']:
            scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        log_string(f'Train Accuracy: {train_instance_acc:.4f}')

        '''测试'''
        with torch.no_grad():
            instance_acc = test(classifier.eval(), testDataLoader, num_class=num_class, device=device)
            log_string(f'Test Accuracy: {instance_acc:.4f}')

            # 保存最佳模型
            if instance_acc > best_acc:
                best_acc = instance_acc
                best_model_path = str(checkpoints_dir) + '/best_model.pth'
                log_string(f'Saving best model to {best_model_path}')

                torch.save({
                    'epoch': epoch,
                    'accuracy': instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'num_class': num_class,
                    'use_normals': args.use_normals,
                    'optimizer_name': training_policy['optimizer'],
                    'scheduler_name': training_policy['scheduler'],
                    'effective_learning_rate': training_policy['effective_learning_rate'],
                    'training_augmentation': augmentation_name,
                }, best_model_path)

    '''最终模型保存'''
    final_model_path = str(checkpoints_dir) + '/final_model.pth'
    log_string(f'Saving final model to {final_model_path}')

    torch.save({
        'epoch': args.epoch,
        'accuracy': instance_acc,
        'model_state_dict': classifier.state_dict(),
        'num_class': num_class,
        'use_normals': args.use_normals,
        'optimizer_name': training_policy['optimizer'],
        'scheduler_name': training_policy['scheduler'],
        'effective_learning_rate': training_policy['effective_learning_rate'],
        'training_augmentation': augmentation_name,
    }, final_model_path)

    log_string(f'Training completed. Best accuracy: {best_acc:.4f}')
    log_string(f'Models saved to: {exp_dir}/checkpoints/')

    # 保存训练配置信息
    config_path = str(checkpoints_dir) + '/training_config.txt'
    with open(config_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Num categories: {args.num_category}\n")
        f.write(f"Epochs: {args.epoch}\n")
        f.write(f"Optimizer: {training_policy['optimizer']}\n")
        f.write(f"Base learning rate: {args.learning_rate}\n")
        f.write(f"Effective learning rate: {training_policy['effective_learning_rate']}\n")
        f.write(f"Scheduler: {training_policy['scheduler']}\n")
        f.write(f"Training augmentation: {augmentation_name}\n")
        f.write(f"Best accuracy: {best_acc:.4f}\n")
        f.write(f"Final accuracy: {instance_acc:.4f}\n")

    log_string(f'Training configuration saved to: {config_path}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
