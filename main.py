import warnings
warnings.filterwarnings("ignore")
import os.path
import sys
from DCMI import *
import torch
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import argparse
from ResNet import resnet18_cbam
from iCIFAR100 import iCIFAR100
from evaluate import Evaluator

parser = argparse.ArgumentParser(description='DCMI')
parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to run')

parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')

parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--start_task', default=1, type=int, help='the number of incremental steps')

parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--g_epoch', default=100, type=int, help='update interval of generator')

parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--exp_des', default='DCMI', type=str, help='experiment description')

args = parser.parse_args()
print(args)


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def setup_data(test_targets, shuffle, seed):
    order = [i for i in range(len(np.unique(test_targets)))]
    if shuffle:
        np.random.seed(seed)
        order = np.random.permutation(len(order)).tolist()
    else:
        order = range(len(order))
    class_order = order
    print(100 * '#')
    print(class_order)
    return map_new_class_index(test_targets, class_order)


def main():
    _set_random(seed=1993)
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(cuda_index)
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.exp_des + '_' + args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size)
    if not os.path.exists('log'):
        os.mkdir('log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename='log/' + file_name + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    feature_extractor = resnet18_cbam()
    # feature_extractor = resnet32()
    model = DCMI_labelAug(args, file_name, feature_extractor, task_size, device)
    # model = DCMI_noSSL(args, file_name, feature_extractor, task_size, device)

    model.setup_data(shuffle=True, seed=1993)

    for i in range(args.task_num+1):
        model.cur_task = i
        model.beforeTrain(i)

        if i < args.start_task:
            model_path = args.save_path + file_name + '/%d_model.pkl' % model.numclass
            prototype = model.model.prototype
            model.model = torch.load(model_path, map_location='cpu').to(model.device)
            model.model.prototype = prototype
            proto_filename = args.save_path + file_name + '/%d_gen.pkl' % model.numclass
            model.netG = torch.load(proto_filename, map_location='cpu').to(model.device)
        else:
            model.train()
        model.afterTrain()

    ####### Test ######
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    evaluator = Evaluator(args.total_nc)
    acc_history = []
    acc_task_history = []

    logging.info("############# Test for up2now Task #############")
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    test_dataset.targets = setup_data(test_dataset.targets, shuffle=True, seed=1993)
    for current_task in range(args.task_num+1):
        evaluator.reset()
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename).to(device)
        model.eval()
        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size)
        for step, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                predicts = model.predict(imgs)
            evaluator.add_batch(labels.cpu().numpy(), predicts.cpu().numpy())

        confusion_matrix, Acc_class, Acc = evaluator.Acc(classes[1])

        # output_filename = args.save_path + file_name + '/confuse_plot_{}.eps'.format(current_task)
        # plt.figure(figsize=(6, 5))
        # plt.imshow(confusion_matrix, cmap='jet') 
        # plt.xlabel('Predicted Class', fontsize=18)
        # plt.ylabel('True Class', fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.colorbar()
        # plt.savefig(output_filename, bbox_inches='tight')
        # plt.show()

        logging.info('Validation:')
        logging.info('[Task: %d, numImages: %5d]' % (current_task, step * args.batch_size + imgs.data.shape[0]))
        logging.info("Acc:{} \n".format(Acc))

        acc_history.append(Acc)
        acc_task = []
        for j in range(current_task + 1):
            if j == 0:
                acc_task.append(np.around(np.mean(Acc_class[:args.fg_nc]), decimals=4))
            else:
                acc_task.append(np.around(np.mean(Acc_class[args.fg_nc + task_size * (
                            j - 1): args.fg_nc + task_size * j]), decimals=4))
        if current_task < args.task_num:
            acc_task.extend((args.task_num - current_task) * [0])
        acc_task_history.append(acc_task)

    acc_task_history = np.array(acc_task_history)
    forget_avg = (np.max(np.array(acc_task_history)[:-1, :-1], axis=0) - np.array(acc_task_history)[-1, :-1]).mean()
    logging.info("Avg_for:{} \n".format(forget_avg))
    logging.info("Avg_acc:{} \n".format(np.mean(acc_history)))
    logging.info("Acc:{} \n".format(list(acc_history)))
    logging.info("Acc_base:{} \n".format(acc_task_history[-1, 0]))
    logging.info("Acc_novel:{} \n".format(np.mean(acc_task_history[-1, 1:])))

if __name__ == "__main__":
    main()