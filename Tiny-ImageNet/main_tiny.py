import os.path
from DCMI_tiny import *
import torch.utils.data
from torch.utils.data import DataLoader
import argparse
from ResNet import resnet18_cbam
from data_manager_tiny import *

parser = argparse.ArgumentParser(description='DCMI')
parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to run')

parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='tiny', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=200, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=100, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--start_task', default=1, type=int, help='resume start task')

parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--g_epoch', default=100, type=float, help='update interval of generator')

parser.add_argument('--gpu', default='1', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--exp_des', default='dcmi', type=str, help='experiment description')

args = parser.parse_args()
print(args)


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main():
    _set_random(1997)
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
    data_manager = DataManager()

    model = DCMI_SupCon(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))

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

    ###### Test ######
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    print("############# Test for each Task #############")
    acc_all = []
    forget_all = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename).to(device)
        model.eval()
        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = class_set[:args.fg_nc]
            else:
                classes = class_set[(args.fg_nc + (i-1)*task_size):(args.fg_nc + i*task_size)]

            test_dataset = data_manager.get_dataset(test_transform, index=classes, train=False)
            test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size)
            correct, total = 0.0, 0.0
            for setp, (imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    predicts = model.predict(imgs)
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        logging.info(acc_up2now)
        if current_task > 0:
            forget = (np.max(np.array(acc_all)[:-1, :current_task], axis=0) - np.array(acc_all)[-1, :current_task]).mean()
            forget_all.append(forget)
    logging.info(acc_all)
    logging.info('forget_avg:{}'.format(forget_all[-1]))

    print("############# Test for up2now Task #############")
    average_acc = 0
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename, map_location='cpu').to(device)
        model.eval()

        classes = class_set[:args.fg_nc+current_task*task_size]
        test_dataset = data_manager.get_dataset(test_transform, index=classes, train=False)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size)
        correct, total = 0.0, 0.0
        for setp, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                predicts = model.predict(imgs)
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        logging.info(accuracy)
        average_acc += accuracy
    logging.info('average acc: ')
    logging.info(average_acc / (args.task_num + 1))


if __name__ == "__main__":
    main()