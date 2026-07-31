import os
import sys
import torch
import numpy as np
import datetime
import logging
import argparse
import random
from pathlib import Path
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from data_utils.ModelNetDataLoader import (
    augment_point_cloud,
)
from data_utils.dataset_config import (
    configure_dataset_args,
    create_backdoor_dataset,
    create_clean_dataset,
)
from model_config import (
    build_training_policy,
    get_region_data_path,
    import_model_module,
)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--model', default='dgcnn',
                        help='model name: dgcnn, pointnet++, or curvenet')
    parser.add_argument('--dataset', type=str, default='modelnet40',
                        help='dataset: modelnet10, modelnet40, or shapenetpart16')
    parser.add_argument('--num_category', default=None, type=int, choices=[10, 16, 40],
                        help='optional category-count validation; inferred from --dataset')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='deprecated; optimizer is selected automatically by model')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum used by DGCNN and CurveNet')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')

    # 后门攻击参数
    parser.add_argument('--poisoned_rate', type=float, default=0.05, help='poison rate')
    parser.add_argument('--target_label', type=int, default=2,
                        help='target label; defaults to 2 for all supported datasets')
    parser.add_argument('--seed', type=int, default=256, help='random seed')

    parser.add_argument('--grid_density', type=float, default=0.4,
                        help='cubic lattice density relative to the median nearest-neighbor spacing')
    parser.add_argument('--attack_region_mode', type=str, default='top2',
                        choices=['top1', 'top2', 'top4', 'top6', 'top8', 'top10', 'top12', 'top14', 'top16',
                                 'bottom1', 'bottom2', 'bottom4',
                                 'random2_connected'],
                        help='attack region selection mode for ablation study')

    parser.add_argument('--region_data_path', type=str,
                        default=None,
                        help='预计算的区域数据路径')

    parser.add_argument('--test_region_data_path', type=str, default=None,
                        help='test-set region pkl path')
    parser.add_argument('--region_data_root', type=str, default='data',
                        help='root for automatically generated region pkl paths')
    parser.add_argument('--data_root', type=str,
                        default=None,
                        help='dataset root; inferred from --dataset when omitted')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader):
    mean_correct = []
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    instance_acc = np.mean(mean_correct)
    return instance_acc



def main(args):
    configure_dataset_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.dataset + '_' + args.model + '_explainable_coords')
    exp_dir.mkdir(exist_ok=True)
    # Compatible with single-region legacy args and current multi-region attack.
    attack_region_idx = getattr(args, 'attack_region_idx', None)
    attack_region_mode = getattr(args, 'attack_region_mode', 'top4')
    if attack_region_idx is None:
        attack_tag = f'{attack_region_mode}'
    else:
        attack_tag = f'{attack_region_mode}_region{attack_region_idx}'
    exp_dir = exp_dir.joinpath(f'{attack_tag}_rate{args.poisoned_rate}')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    num_class = args.num_category

    # 构建区域数据路径
    region_data_path = args.region_data_path or str(get_region_data_path(
        args.model,
        args.num_category,
        args.dataset,
        'train',
        root=args.region_data_root,
        num_regions=16,
    ))
    args.region_data_path = region_data_path
    if not os.path.exists(region_data_path):
        # 尝试自动构建路径
        base_name = os.path.basename(region_data_path)
        if 'train' in base_name:
            region_data_path = str(get_region_data_path(
                args.model,
                args.num_category,
                args.dataset,
                'train',
                root=args.region_data_root,
                num_regions=16,
            ))
        else:
            region_data_path = str(get_region_data_path(
                args.model,
                args.num_category,
                args.dataset,
                'test',
                root=args.region_data_root,
                num_regions=16,
            ))

        args.region_data_path = region_data_path

    test_region_data_path = args.test_region_data_path or str(
        Path(region_data_path).parent
        / f'{args.dataset}_test_regions_with_points.pkl'
    )
    args.test_region_data_path = test_region_data_path
    test_args = argparse.Namespace(**vars(args))
    test_args.region_data_path = test_region_data_path

    train_dataset = create_backdoor_dataset(args, split='train')
    test_dataset = create_clean_dataset(
        test_args,
        split='test',
        process_data=False,
    )
    test_bd_dataset = create_backdoor_dataset(test_args, split='test')

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0)
    testbdDataLoader = torch.utils.data.DataLoader(test_bd_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=0)

    '''MODEL LOADING'''
    model = import_model_module(args.model)
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()

    log_string('---Training from begin.')

    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

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

    start_epoch = 0
    global_epoch = 0
    global_step = 0

    '''TRANING'''
    logger.info('Start training...')
    best_acc = 0.0
    best_asr = 0.0

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        if not training_policy['scheduler_step_after_epoch']:
            scheduler.step()

        for _, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                                     smoothing=0.9):
            optimizer.zero_grad()
            points = points.numpy()

            points = augment_point_cloud(points)
            # 转回 tensor
            points = torch.tensor(points, dtype=torch.float32)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points = points.cuda()
                target = target.cuda()

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
            global_step += 1

        if training_policy['scheduler_step_after_epoch']:
            scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        with torch.no_grad():
            # 测试干净准确率
            instance_acc = test(classifier.eval(), testDataLoader)
            # 测试后门攻击成功率
            instance_bd_acc = test(classifier.eval(), testbdDataLoader)

            log_string('Test Instance Accuracy: %f' % instance_acc)
            log_string('Backdoor Test Instance Accuracy: %f' % instance_bd_acc)

            if instance_acc > best_acc:
                best_acc = instance_acc

            if instance_bd_acc > best_asr:
                best_asr = instance_bd_acc
                logger.info('Save best backdoor model...')

                savepath = str(checkpoints_dir) + '/best_backdoor_model.pth'
                log_string('Saving best attack model (ASR: %.4f) at %s' % (best_asr, savepath))
                state = {
                    'instance_acc': instance_acc,
                    'instance_bd_acc': instance_bd_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_name': training_policy['optimizer'],
                    'scheduler_name': training_policy['scheduler'],
                    'effective_learning_rate': training_policy['effective_learning_rate'],
                    'epoch': epoch,
                }
                torch.save(state, savepath)

            # 保存最后模型
            if epoch == args.epoch - 1:
                logger.info('Save final model...')
                savepath = str(checkpoints_dir) + '/final_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'instance_acc': instance_acc,
                    'instance_bd_acc': instance_bd_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_name': training_policy['optimizer'],
                    'scheduler_name': training_policy['scheduler'],
                    'effective_learning_rate': training_policy['effective_learning_rate'],
                    'epoch': epoch,
                }
                torch.save(state, savepath)

            global_epoch += 1

    log_string('Best Clean Accuracy: %f' % best_acc)
    log_string('Best Attack Success Rate: %f' % best_asr)
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
