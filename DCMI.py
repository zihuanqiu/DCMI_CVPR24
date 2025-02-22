import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from myNetwork import network
from iCIFAR100 import iCIFAR100
from generator import CondGenerator, Discriminator, cacl_gradient_penalty
from tqdm import tqdm
import logging
import os
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy
import math
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale, Normalize, Denormalize
from kornia.augmentation.container import AugmentationSequential

GAMMA=10

class DCMI_noSSL:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc, feature_extractor)
        self.netG = CondGenerator(num_classes=args.fg_nc).cuda()
        self.cur_task = 0
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None

    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        print(100*'#')
        print(self.class_order)

        self.train_dataset.targets = self.map_new_class_index(train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index(test_targets, self.class_order)

    def beforeTrain(self, current_task):
        self.model.eval()
        increment = [self.args.fg_nc] + [self.task_size]*current_task
        self.model.increment = increment

        if current_task == 0:
            classes = [0, self.numclass]
            self.model.class2task = np.array([0]*self.numclass)
            self.old_class = 0
        else:
            self.numclass += self.task_size
            self.old_class = self.numclass - self.task_size
            classes = [self.numclass-self.task_size, self.numclass]
            self.model.class2task = np.concatenate([self.model.class2task, np.array([current_task] * self.task_size)])
            self.model.incremental_learning(self.task_size)
            self.netG = CondGenerator(num_classes=self.old_class).cuda()
            self.netD = Discriminator().cuda()
            self.model.prototype.append([])
            self.old_proto = torch.tensor(np.concatenate(self.model.prototype[:-1])[:self.old_class]).cuda()
            self.sim_mat = torch.mm(F.normalize(self.old_proto), F.normalize(self.old_proto).T)

        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        self.model.to(self.device)
        self.model.train()

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size,
                                  pin_memory=True,
                                  num_workers=8,
                                  drop_last=True)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size,
                                 pin_memory=True,
                                 num_workers=8)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader


    def train(self):
        opt_params = self.model.parameters()

        if self.old_model is not None:
            self._update_gnet()
            opt_params = nn.ModuleList([self.model.fc, self.model.feature.layer4]).parameters()

        opt = torch.optim.Adam(opt_params, lr=self.learning_rate, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(opt, self.epochs)
        prog_bar = tqdm(range(self.epochs))
        loss_g = 0
        for epoch in prog_bar:
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                loss = self._update_representation(images, target, opt)

            if epoch % self.args.print_freq == 0:
                self.protoSave()
                accuracy = self._test(self.test_loader)
            info = 'Task {}, Epoch{}/{} => Loss {:.3f}, Loss G {:.3f}, Test Acc:{:.3f}' \
                .format(self.cur_task, epoch + 1, self.epochs, loss, loss_g, accuracy)
            prog_bar.set_description(info)
            scheduler.step()
        logging.info(info)

    def _update_gnet(self):
        self.netG.train()
        self.netD.train()

        opt_g = torch.optim.Adam(self.netG.parameters(), lr=1e-3, betas=[0., 0.9])
        sch_opt_g = CosineAnnealingLR(opt_g, self.args.g_epoch)

        opt_d = torch.optim.Adam(self.netD.parameters(), lr=1e-3, betas=[0., 0.9])
        sch_opt_d = CosineAnnealingLR(opt_d, self.args.g_epoch)
        prog_bar = tqdm(range(self.args.g_epoch))

        for epoch in prog_bar:
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                prob = torch.rand(1)

                if prob < 0.6:
                    # 1.
                    r_out_old = self.old_model(images)
                    linear = F.softmax(torch.mm(F.normalize(r_out_old['feat']), F.normalize(self.old_proto).T) * GAMMA, dim=1)

                    for _ in range(5):
                        mix_emb = torch.mm(linear, self.netG.emb.weight)
                        z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
                        fake_img = self.netG(z, mix_emb)

                        d_out_fake = self.netD(fake_img.detach())
                        d_out_real = self.netD(images.detach())

                        loss_gp = cacl_gradient_penalty(self.netD, images.detach(), fake_img.detach())
                        loss_d = d_out_fake.mean() - d_out_real.mean() + loss_gp*10

                        opt_d.zero_grad()
                        loss_d.backward()
                        opt_d.step()
                else:
                    # 2.
                    old_label = torch.randint(0, self.old_class,( self.args.batch_size,), device=self.device)
                    linear = F.softmax(self.sim_mat[old_label.long()] * GAMMA, dim=1)

                mix_emb = torch.mm(linear, self.netG.emb.weight)
                z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
                fake_img = self.netG(z, mix_emb)

                out_old = self.old_model(fake_img)

                log_sim = F.log_softmax(torch.mm(F.normalize(out_old['feat']), F.normalize(self.old_proto).T) * GAMMA, dim=1)

                loss_align = (-(linear * log_sim).sum(dim=1)).mean()

                d_out_fake = self.netD(fake_img)
                loss_local = -d_out_fake.mean()

                loss_g = loss_align*10 + loss_local*0.5

                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

            sch_opt_g.step()
            sch_opt_d.step()
            info = 'Train G => round {}/{} => loss_g {:.3f}, loss_d {:.3f}'.format(epoch + 1, len(prog_bar), loss_g.item(), loss_d.item())
            prog_bar.set_description(info)

            # save_image_batch(normalize(images, reverse=True), save_path + "r_{}.png".format(epoch))
            # save_image_batch(normalize(fake_img, reverse=True), save_path + "f_{}.png".format(epoch))

        self.netG.eval()

    def _update_representation(self, imgs, target, opt):
        if self.old_model is None:

            out = self.model(imgs)
            loss_cls = nn.CrossEntropyLoss()(out['logits'][0] / self.args.temp, target)

            loss = loss_cls
            opt.zero_grad()
            loss.backward()
            opt.step()

            return loss.item()
        else:
            self.model.eval()
            self.model.feature.layer4.train()
            old_label = torch.randint(0, self.old_class, (self.args.batch_size,), device=self.device)

            sim_ = self.sim_mat[old_label.long()]
            linear = F.softmax(sim_ * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)

            fake = self.netG(z, mix_emb)

            images = torch.cat([fake.detach(), imgs])
            out = self.model(images)

            with torch.no_grad():
                out_old = self.old_model(images)

            kd_loss = torch.stack([torch.norm(logit_o-logit_n, dim=1).mean()
                                   for logit_o, logit_n in
                                   zip(out['logits'][:-1], out_old['logits'])]).mean()

            loss_cls = F.cross_entropy(out['logits'][-1][-self.args.batch_size:] / self.args.temp, target-self.old_class)

            loss_em = 1 - F.cosine_similarity(out['feat'], out_old['feat']).mean()

            loss = loss_cls + loss_em*10 + kd_loss*10

            opt.zero_grad()
            loss.backward()
            opt.step()

            return loss.item()

    @torch.no_grad()
    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            predicts = self.model.predict(imgs)
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct / total
        self.model.train()
        return accuracy

    def afterTrain(self):
        self.protoSave()
        if self.cur_task >= self.args.start_task:
            path = self.args.save_path + self.file_name + '/'
            model_filename = path + '%d_model.pkl' % self.numclass
            gen_filename = path + '%d_gen.pkl' % self.numclass
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(self.model, model_filename)
            torch.save(self.netG, gen_filename)
        if self.cur_task > 0:
            self.imgSave()
        accuracy = self._test(self.test_loader)
        info = 'Task {}, Test_acc {:.3f}'.format(self.cur_task, accuracy)
        logging.info(info)
        self.old_model = deepcopy(self.model)
        self.old_model.to(self.device)
        self.old_model.eval()

    @torch.no_grad()
    def imgSave(self):
        save_path = self.args.save_path + self.file_name + '/save_img_task{}'.format(self.cur_task)
        for i in range(self.old_class, self.numclass):
            img = self.test_dataset.get_test_image_class(i, 36)
            img = normalize(img, reverse=True)
            save_image_batch(img, save_path + "/cls_{}.png".format(i))

        for i in range(self.old_class):
            y = i*torch.ones(36, device=self.device)
            linear = F.softmax(self.sim_mat[y.long()] * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)

            vis_img = self.netG(z, mix_emb)
            vis_img = normalize(vis_img, reverse=True)
            save_image_batch(vis_img, save_path + "/fake_cls_{}.png".format(i))

            img = self.test_dataset.get_test_image_class(i, 36)
            img = normalize(img, reverse=True)
            save_image_batch(img, save_path + "/cls_{}.png".format(i))

    def protoSave(self):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(self.train_loader):
                feature = self.model.feature(images.to(self.device))
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        labels = np.concatenate(labels)
        labels_set = np.unique(labels)
        features = np.concatenate(features)

        prototype = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]

            prototype.append(np.mean(feature_classwise, axis=0))

        self.model.prototype[-1] = np.stack(prototype)

    @torch.no_grad()
    def compute_test_features(self):
        self.old_model.eval()
        features, lbs = [], []
        tqdm_batch = tqdm(
            total=len(self.test_loader), desc=f"[Compute test features]"
        )
        for batch, (_, images, target) in enumerate(self.test_loader):
            images, target = images.to(self.device), target.to(self.device)
            features.append(self.old_model.feature(images))
            lbs.append(target)
            tqdm_batch.update()
        tqdm_batch.close()
        features = torch.cat(features)
        lbs = torch.cat(lbs)
        return features, lbs

    @torch.no_grad()
    def teaser_visualize(self, selected_old_class):
        old_proto = torch.tensor(np.concatenate(self.model.prototype[:-1])).cuda()

        feat, lbs = self.compute_test_features()

        selected_old_proto = [old_proto[i] for i in selected_old_class]
        selected_old_feat = [feat[lbs==i] for i in selected_old_class]

        selected_old_proto = torch.stack(selected_old_proto)
        selected_old_feat = torch.cat(selected_old_feat)

        fake_old_feat = []
        for i in selected_old_class:
            y = i*torch.ones(100, device=self.device)

            sim_ = self.sim_mat[y.long()]
            linear = F.softmax(sim_ * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
            fake_old_feat.append(self.old_model.feature(self.netG(z, mix_emb)))

        fake_old_feat = torch.cat(fake_old_feat)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        vectors = np.concatenate([fake_old_feat.cpu().numpy(), selected_old_feat.cpu().numpy(), selected_old_proto.cpu().numpy()])
        embeddings = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=1, metric="cosine").fit_transform(vectors)

        vis_fx = embeddings[:fake_old_feat.shape[0], 0]
        vis_fy = embeddings[:fake_old_feat.shape[0], 1]

        vis_fx = np.split(vis_fx, len(selected_old_class))
        vis_fy = np.split(vis_fy, len(selected_old_class))

        vis_x = embeddings[fake_old_feat.shape[0]:-len(selected_old_class), 0]
        vis_y = embeddings[fake_old_feat.shape[0]:-len(selected_old_class), 1]

        vis_x = np.split(vis_x, len(selected_old_class))
        vis_y = np.split(vis_y, len(selected_old_class))

        old_proto_x = embeddings[-len(selected_old_class):, 0]
        old_proto_y = embeddings[-len(selected_old_class):, 1]

        plt.figure(figsize=(5, 5))
        color = ['c', 'b', 'g', 'r', 'm', 'y']
        for i, (x, y) in enumerate(zip(vis_fx, vis_fy)):
            plt.scatter(x, y, c=color[i], s=10, marker='o')

        for i, (x, y) in enumerate(zip(vis_x, vis_y)):
            plt.scatter(x, y, c=color[i], s=10, marker='x')

        plt.scatter(old_proto_x, old_proto_y, c='k', marker='^', s=50)
        plt.xticks([])
        plt.yticks([])

        save_path = self.args.save_path + self.file_name + "/teaser_{}.eps".format(self.cur_task)
        base_dir = os.path.dirname(save_path)
        if base_dir != '':
            os.makedirs(base_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')


class DCMI_Supcon:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc, feature_extractor)
        self.netG = CondGenerator(num_classes=args.fg_nc).cuda()
        self.cur_task = 0
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.transform = AugmentationSequential(
            Denormalize(mean=torch.tensor([0.5071, 0.4867, 0.4408]),
                        std=torch.tensor([0.2675, 0.2565, 0.2761])),
            RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            RandomGrayscale(p=0.2),
            Normalize(mean=torch.tensor([0.5071, 0.4867, 0.4408]),
                      std=torch.tensor([0.2675, 0.2565, 0.2761])),
        )

        self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None

    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        print(100*'#')
        print(self.class_order)

        self.train_dataset.targets = self.map_new_class_index(train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index(test_targets, self.class_order)

    def beforeTrain(self, current_task):
        self.model.eval()
        increment = [self.args.fg_nc] + [self.task_size]*current_task
        self.model.increment = increment

        if current_task == 0:
            classes = [0, self.numclass]
            self.model.class2task = np.array([0]*self.numclass)
            self.old_class = 0
        else:
            self.numclass += self.task_size
            self.old_class = self.numclass - self.task_size
            classes = [self.numclass-self.task_size, self.numclass]
            self.model.class2task = np.concatenate([self.model.class2task, np.array([current_task] * self.task_size)])
            self.model.incremental_learning(self.task_size)
            self.netG = CondGenerator(num_classes=self.old_class).cuda()
            self.netD = Discriminator().cuda()
            self.model.prototype.append([])
            self.old_proto = torch.tensor(np.concatenate(self.model.prototype[:-1])[:self.old_class]).cuda()
            self.sim_mat = torch.mm(F.normalize(self.old_proto), F.normalize(self.old_proto).T)

        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        self.model.to(self.device)
        self.model.train()

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size,
                                  pin_memory=True,
                                  num_workers=8,
                                  drop_last=True)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size,
                                 pin_memory=True,
                                 num_workers=8)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader


    def train(self):
        opt_params = self.model.parameters()

        if self.old_model is not None:
            self._update_gnet()
            opt_params = nn.ModuleList([self.model.fc, self.model.feature.layer4]).parameters()

        opt = torch.optim.Adam(opt_params, lr=self.learning_rate, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(opt, self.epochs)
        prog_bar = tqdm(range(self.epochs))
        loss_g = 0
        for epoch in prog_bar:
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                loss = self._update_representation(images, target, opt)

            if epoch % self.args.print_freq == 0:
                self.protoSave()
                accuracy = self._test(self.test_loader)
            info = 'Task {}, Epoch{}/{} => Loss {:.3f}, Loss G {:.3f}, Test Acc:{:.3f}' \
                .format(self.cur_task, epoch + 1, self.epochs, loss, loss_g, accuracy)
            prog_bar.set_description(info)
            scheduler.step()
        logging.info(info)

    def _update_gnet(self):
        self.netG.train()
        self.netD.train()

        opt_g = torch.optim.Adam(self.netG.parameters(), lr=1e-3, betas=[0., 0.9])
        sch_opt_g = CosineAnnealingLR(opt_g, self.args.g_epoch)

        opt_d = torch.optim.Adam(self.netD.parameters(), lr=1e-3, betas=[0., 0.9])
        sch_opt_d = CosineAnnealingLR(opt_d, self.args.g_epoch)
        prog_bar = tqdm(range(self.args.g_epoch))
        save_path = self.args.save_path + self.file_name + '/save_img_task{}/cache/'.format(self.cur_task)

        for epoch in prog_bar:
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                prob = torch.rand(1)

                if prob < 0.6:
                    # 1.
                    r_out_old = self.old_model(images)
                    linear = F.softmax(torch.mm(F.normalize(r_out_old['feat']), F.normalize(self.old_proto).T) * GAMMA, dim=1)

                    for _ in range(5):
                        mix_emb = torch.mm(linear, self.netG.emb.weight)
                        z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
                        fake_img = self.netG(z, mix_emb)

                        d_out_fake = self.netD(fake_img.detach())
                        d_out_real = self.netD(images.detach())

                        loss_gp = cacl_gradient_penalty(self.netD, images.detach(), fake_img.detach())
                        loss_d = d_out_fake.mean() - d_out_real.mean() + loss_gp*10

                        opt_d.zero_grad()
                        loss_d.backward()
                        opt_d.step()
                else:
                    # 2.
                    old_label = torch.randint(0, self.old_class,( self.args.batch_size,), device=self.device)
                    linear = F.softmax(self.sim_mat[old_label.long()] * GAMMA, dim=1)

                mix_emb = torch.mm(linear, self.netG.emb.weight)
                z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
                fake_img = self.netG(z, mix_emb)

                out_old = self.old_model(fake_img)

                log_sim = F.log_softmax(torch.mm(F.normalize(out_old['feat']), F.normalize(self.old_proto).T) * GAMMA, dim=1)

                loss_align = (-(linear * log_sim).sum(dim=1)).mean()

                d_out_fake = self.netD(fake_img)
                loss_local = -d_out_fake.mean()

                loss_g = loss_align*10 + loss_local*0.5

                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

            sch_opt_g.step()
            sch_opt_d.step()
            info = 'Train G => round {}/{} => loss_g {:.3f}, loss_d {:.3f}'.format(epoch + 1, len(prog_bar), loss_g.item(), loss_d.item())
            prog_bar.set_description(info)

            save_image_batch(normalize(images, reverse=True), save_path + "r_{}.png".format(epoch))
            save_image_batch(normalize(fake_img, reverse=True), save_path + "f_{}.png".format(epoch))

        self.netG.eval()

    def _update_representation(self, imgs, target, opt):
        if self.old_model is None:
            out = self.model(torch.cat([imgs, self.transform(imgs)]))
            loss_cls = nn.CrossEntropyLoss()(out['logits'][0] / self.args.temp, torch.cat([target, target]))
            loss_con = self.model.SimSiam(out['proj'][:target.size(0)],
                                          out['proj'][target.size(0):])

            loss = loss_cls + loss_con

            opt.zero_grad()
            loss.backward()
            opt.step()

            return loss.item()
        else:
            self.model.eval()
            self.model.feature.layer4.train()

            old_label = torch.randint(0, self.old_class, (self.args.batch_size,), device=self.device)

            sim_ = self.sim_mat[old_label.long()]
            linear = F.softmax(sim_ * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)

            fake = self.netG(z, mix_emb)

            images = torch.cat([fake.detach(), imgs])
            out = self.model(images)

            with torch.no_grad():
                out_old = self.old_model(images)

            kd_loss = torch.stack([torch.norm(logit_o-logit_n, dim=1).mean()
                                   for logit_o, logit_n in
                                   zip(out['logits'][:-1], out_old['logits'])]).mean()

            loss_cls = F.cross_entropy(out['logits'][-1][-self.args.batch_size:] / self.args.temp, target-self.old_class)

            loss_em = 1 - F.cosine_similarity(out['feat'][:-self.args.batch_size], out_old['feat'][:-self.args.batch_size]).mean()

            loss = loss_cls + loss_em*10 + kd_loss*10

            opt.zero_grad()
            loss.backward()
            opt.step()

            return loss.item()

    @torch.no_grad()
    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            predicts = self.model.predict(imgs)
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct / total
        self.model.train()
        return accuracy

    def afterTrain(self):
        self.protoSave()
        if self.cur_task >= self.args.start_task:
            path = self.args.save_path + self.file_name + '/'
            model_filename = path + '%d_model.pkl' % self.numclass
            gen_filename = path + '%d_gen.pkl' % self.numclass
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(self.model, model_filename)
            torch.save(self.netG, gen_filename)
        if self.cur_task > 0:
            self.teaser_visualize([1, 4, 14, 24, 35])
            self.imgSave()
        accuracy = self._test(self.test_loader)
        info = 'Task {}, Test_acc {:.3f}'.format(self.cur_task, accuracy)
        logging.info(info)
        self.old_model = deepcopy(self.model)
        self.old_model.to(self.device)
        self.old_model.eval()

    @torch.no_grad()
    def imgSave(self):
        save_path = self.args.save_path + self.file_name + '/save_img_task{}'.format(self.cur_task)
        for i in range(self.old_class, self.numclass):
            img = self.test_dataset.get_test_image_class(i, 36)
            img = normalize(img, reverse=True)
            save_image_batch(img, save_path + "/cls_{}.png".format(i))

        for i in range(self.old_class):
            y = i*torch.ones(36, device=self.device)
            linear = F.softmax(self.sim_mat[y.long()] * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)

            vis_img = self.netG(z, mix_emb)
            vis_img = normalize(vis_img, reverse=True)
            save_image_batch(vis_img, save_path + "/fake_cls_{}.png".format(i))

            img = self.test_dataset.get_test_image_class(i, 36)
            img = normalize(img, reverse=True)
            save_image_batch(img, save_path + "/cls_{}.png".format(i))

    def protoSave(self):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(self.train_loader):
                feature = self.model.feature(images.to(self.device))
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        labels = np.concatenate(labels)
        labels_set = np.unique(labels)
        features = np.concatenate(features)

        prototype = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]

            prototype.append(np.mean(feature_classwise, axis=0))

        self.model.prototype[-1] = np.stack(prototype)

    @torch.no_grad()
    def compute_test_features(self):
        self.old_model.eval()
        features, lbs = [], []
        tqdm_batch = tqdm(
            total=len(self.test_loader), desc=f"[Compute test features]"
        )
        for batch, (_, images, target) in enumerate(self.test_loader):
            images, target = images.to(self.device), target.to(self.device)
            features.append(self.old_model.feature(images))
            lbs.append(target)
            tqdm_batch.update()
        tqdm_batch.close()
        features = torch.cat(features)
        lbs = torch.cat(lbs)
        return features, lbs

    @torch.no_grad()
    def teaser_visualize(self, selected_old_class):
        old_proto = torch.tensor(np.concatenate(self.model.prototype[:-1])).cuda()

        feat, lbs = self.compute_test_features()

        selected_old_proto = [old_proto[i] for i in selected_old_class]
        selected_old_feat = [feat[lbs==i] for i in selected_old_class]

        selected_old_proto = torch.stack(selected_old_proto)
        selected_old_feat = torch.cat(selected_old_feat)

        fake_old_feat = []
        for i in selected_old_class:
            y = i*torch.ones(100, device=self.device)

            sim_ = self.sim_mat[y.long()]
            linear = F.softmax(sim_ * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
            fake_old_feat.append(self.old_model.feature(self.netG(z, mix_emb)))

        fake_old_feat = torch.cat(fake_old_feat)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        vectors = np.concatenate([fake_old_feat.cpu().numpy(), selected_old_feat.cpu().numpy(), selected_old_proto.cpu().numpy()])
        embeddings = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=1, metric="cosine").fit_transform(vectors)

        vis_fx = embeddings[:fake_old_feat.shape[0], 0]
        vis_fy = embeddings[:fake_old_feat.shape[0], 1]

        vis_fx = np.split(vis_fx, len(selected_old_class))
        vis_fy = np.split(vis_fy, len(selected_old_class))

        vis_x = embeddings[fake_old_feat.shape[0]:-len(selected_old_class), 0]
        vis_y = embeddings[fake_old_feat.shape[0]:-len(selected_old_class), 1]

        vis_x = np.split(vis_x, len(selected_old_class))
        vis_y = np.split(vis_y, len(selected_old_class))

        old_proto_x = embeddings[-len(selected_old_class):, 0]
        old_proto_y = embeddings[-len(selected_old_class):, 1]

        plt.figure(figsize=(5, 5))
        color = ['c', 'b', 'g', 'r', 'm', 'y']
        for i, (x, y) in enumerate(zip(vis_fx, vis_fy)):
            plt.scatter(x, y, c=color[i], s=10, marker='o')

        for i, (x, y) in enumerate(zip(vis_x, vis_y)):
            plt.scatter(x, y, c=color[i], s=10, marker='x')

        plt.scatter(old_proto_x, old_proto_y, c='k', marker='^', s=50)
        plt.xticks([])
        plt.yticks([])

        save_path = self.args.save_path + self.file_name + "/teaser_{}.eps".format(self.cur_task)
        base_dir = os.path.dirname(save_path)
        if base_dir != '':
            os.makedirs(base_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')


class DCMI_labelAug:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc*4, feature_extractor)
        self.netG = CondGenerator(num_classes=args.fg_nc).cuda()
        self.cur_task = 0
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None

    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        print(100*'#')
        print(self.class_order)

        self.train_dataset.targets = self.map_new_class_index(train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index(test_targets, self.class_order)

    def beforeTrain(self, current_task):
        self.model.eval()
        increment = [self.args.fg_nc] + [self.task_size]*current_task
        self.model.increment = increment

        if current_task == 0:
            classes = [0, self.numclass]
            self.model.class2task = np.array([0]*self.numclass)
            self.old_class = 0
        else:
            self.numclass += self.task_size
            self.old_class = self.numclass - self.task_size
            classes = [self.numclass-self.task_size, self.numclass]
            self.model.class2task = np.concatenate([self.model.class2task, np.array([current_task] * self.task_size)])
            self.model.incremental_learning(self.task_size)
            self.netG = CondGenerator(num_classes=self.old_class).cuda()
            self.netD = Discriminator().cuda()
            self.model.prototype.append([])
            self.old_proto = torch.tensor(np.concatenate(self.model.prototype[:-1])[:self.old_class]).cuda()
            self.sim_mat = torch.mm(F.normalize(self.old_proto), F.normalize(self.old_proto).T)

        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        self.model.to(self.device)
        self.model.train()

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size,
                                  pin_memory=True,
                                  num_workers=8,
                                  drop_last=True)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size,
                                 pin_memory=True,
                                 num_workers=8)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self):
        opt_params = self.model.parameters()

        if self.old_model is not None:
            self._update_gnet()
            opt_params = nn.ModuleList([self.model.fc, self.model.feature.layer4]).parameters()

        opt = torch.optim.Adam(opt_params, lr=self.learning_rate, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(opt, self.epochs)
        prog_bar = tqdm(range(self.epochs))
        loss_g = 0
        for epoch in prog_bar:
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                loss = self._update_representation(images, target, opt)

            if epoch % self.args.print_freq == 0:
                self.protoSave()
                accuracy = self._test(self.test_loader)  # TODO: fix label aug test in the first task
            info = 'Task {}, Epoch{}/{} => Loss {:.3f}, Loss G {:.3f}, Test Acc:{:.3f}' \
                .format(self.cur_task, epoch + 1, self.epochs, loss, loss_g, accuracy)
            prog_bar.set_description(info)
            scheduler.step()
        logging.info(info)

    def _update_gnet(self):
        self.netG.train()
        self.netD.train()

        opt_g = torch.optim.Adam(self.netG.parameters(), lr=1e-3, betas=[0., 0.9])
        sch_opt_g = CosineAnnealingLR(opt_g, self.args.g_epoch)

        opt_d = torch.optim.Adam(self.netD.parameters(), lr=1e-3, betas=[0., 0.9])
        sch_opt_d = CosineAnnealingLR(opt_d, self.args.g_epoch)
        prog_bar = tqdm(range(self.args.g_epoch))
        save_path = self.args.save_path + self.file_name + '/save_img_task{}/cache/'.format(self.cur_task)

        for epoch in prog_bar:
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                old_label = torch.randint(0, self.old_class, (self.args.batch_size,), device=self.device)
                linear = F.softmax(self.sim_mat[old_label.long()] * GAMMA, dim=1)

                for _ in range(5):
                    mix_emb = torch.mm(linear, self.netG.emb.weight)
                    z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
                    fake_img = self.netG(z, mix_emb)

                    d_out_fake = self.netD(fake_img.detach())
                    d_out_real = self.netD(images.detach())

                    loss_gp = cacl_gradient_penalty(self.netD, images.detach(), fake_img.detach())
                    loss_d = d_out_fake.mean() - d_out_real.mean() + loss_gp*10

                    opt_d.zero_grad()
                    loss_d.backward()
                    opt_d.step()

                mix_emb = torch.mm(linear, self.netG.emb.weight)
                z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
                fake_img = self.netG(z, mix_emb)

                out_old = self.old_model(fake_img)

                log_sim = F.log_softmax(torch.mm(F.normalize(out_old['feat']), F.normalize(self.old_proto).T) * GAMMA, dim=1)

                loss_align = (-(linear * log_sim).sum(dim=1)).mean()

                d_out_fake = self.netD(fake_img)
                loss_local = -d_out_fake.mean()

                loss_g = loss_align*10 + loss_local*0.5

                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

            sch_opt_g.step()
            sch_opt_d.step()
            info = 'Train G => round {}/{} => loss_g {:.3f}, loss_d {:.3f}'.format(epoch + 1, len(prog_bar), loss_g.item(), loss_d.item())
            prog_bar.set_description(info)

            save_image_batch(normalize(images, reverse=True), save_path + "r_{}.png".format(epoch))
            save_image_batch(normalize(fake_img, reverse=True), save_path + "f_{}.png".format(epoch))

        self.netG.eval()

    def _update_representation(self, imgs, target, opt):
        if self.old_model is None:
            # self-supervised learning based label augmentation
            images = torch.stack([torch.rot90(imgs, k, (2, 3)) for k in range(4)], 1)
            imgs = images.view(-1, 3, 32, 32)
            target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

            out = self.model(imgs)
            loss_cls = nn.CrossEntropyLoss()(out['logits'][0] / self.args.temp, target)

            loss = loss_cls
            opt.zero_grad()
            loss.backward()
            opt.step()

            return loss.item()
        else:
            self.model.eval()
            self.model.feature.layer4.train()

            old_label = torch.randint(0, self.old_class, (self.args.batch_size,), device=self.device)

            sim_ = self.sim_mat[old_label.long()]
            linear = F.softmax(sim_ * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)

            fake = self.netG(z, mix_emb)

            images = torch.cat([fake.detach(), imgs])
            out = self.model(images)

            with torch.no_grad():
                out_old = self.old_model(images)

            kd_loss = torch.stack([torch.norm(logit_o-logit_n, dim=1).mean()
                                   for logit_o, logit_n in
                                   zip(out['logits'][:-1], out_old['logits'])]).mean()

            loss_cls = F.cross_entropy(out['logits'][-1][-self.args.batch_size:] / self.args.temp, target-self.old_class)

            loss_em = 1 - F.cosine_similarity(out['feat'][:-self.args.batch_size], out_old['feat'][:-self.args.batch_size]).mean()

            loss = loss_cls + loss_em*10 + kd_loss*10

            opt.zero_grad()
            loss.backward()
            opt.step()

            return loss.item()

    @torch.no_grad()
    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            predicts = self.model.predict(imgs)
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct / total
        self.model.train()
        return accuracy

    def afterTrain(self):
        self.protoSave()
        if self.cur_task >= self.args.start_task:
            if self.cur_task == self.args.start_task == 0:
                self.model.mergeCLF()
            path = self.args.save_path + self.file_name + '/'
            model_filename = path + '%d_model.pkl' % self.numclass
            gen_filename = path + '%d_gen.pkl' % self.numclass
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(self.model, model_filename)
            torch.save(self.netG, gen_filename)
        if self.cur_task >= 1:
            self.teaser_visualize([1, 4, 14, 24, 35])
            self.imgSave()
        accuracy = self._test(self.test_loader)
        info = 'Task {}, Test_acc {:.3f}'.format(self.cur_task, accuracy)
        logging.info(info)
        self.old_model = deepcopy(self.model)
        self.old_model.to(self.device)
        self.old_model.eval()

    @torch.no_grad()
    def imgSave(self):
        save_path = self.args.save_path + self.file_name + '/save_img_task{}'.format(self.cur_task)
        for i in range(self.old_class, self.numclass):
            img = self.test_dataset.get_test_image_class(i, 36)
            img = normalize(img, reverse=True)
            save_image_batch(img, save_path + "/cls_{}.png".format(i))

        for i in range(self.old_class):
            y = i*torch.ones(36, device=self.device)
            linear = F.softmax(self.sim_mat[y.long()] * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)

            vis_img = self.netG(z, mix_emb)
            vis_img = normalize(vis_img, reverse=True)
            save_image_batch(vis_img, save_path + "/fake_cls_{}.png".format(i))


            img = self.test_dataset.get_test_image_class(i, 36)
            img = normalize(img, reverse=True)
            save_image_batch(img, save_path + "/cls_{}.png".format(i))

    @torch.no_grad()
    def protoSave(self):
        features = []
        labels = []
        self.model.eval()
        for i, (indexs, images, target) in enumerate(self.train_loader):
            feature = self.model.feature(images.to(self.device))
            labels.append(target.numpy())
            features.append(feature.cpu().numpy())
        labels = np.concatenate(labels)
        labels_set = np.unique(labels)
        features = np.concatenate(features)

        prototype = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]

            prototype.append(np.mean(feature_classwise, axis=0))

        self.model.prototype[-1] = np.stack(prototype)

    @torch.no_grad()
    def compute_test_features(self):
        self.old_model.eval()
        features, lbs = [], []
        tqdm_batch = tqdm(
            total=len(self.test_loader), desc=f"[Compute test features]"
        )
        for batch, (_, images, target) in enumerate(self.test_loader):
            images, target = images.to(self.device), target.to(self.device)
            features.append(self.old_model.feature(images))
            lbs.append(target)
            tqdm_batch.update()
        tqdm_batch.close()
        features = torch.cat(features)
        lbs = torch.cat(lbs)
        return features, lbs

    @torch.no_grad()
    def teaser_visualize(self, selected_old_class):
        feat, lbs = self.compute_test_features()

        selected_old_feat = [feat[lbs==i] for i in selected_old_class]

        selected_old_feat = torch.cat(selected_old_feat)

        fake_old_feat = []
        for i in selected_old_class:
            y = i*torch.ones(100, device=self.device)

            sim_ = self.sim_mat[y.long()]
            linear = F.softmax(sim_ * GAMMA, dim=1)

            mix_emb = torch.mm(linear, self.netG.emb.weight)
            z = torch.randn(size=(mix_emb.size(0), 100), device=mix_emb.device)
            fake_old_feat.append(self.old_model.feature(self.netG(z, mix_emb)))

        fake_old_feat = torch.cat(fake_old_feat)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        vectors = np.concatenate([fake_old_feat.cpu().numpy(), selected_old_feat.cpu().numpy()])
        embeddings = TSNE(n_components=2, learning_rate='auto', random_state=1, metric="cosine", perplexity=10).fit_transform(vectors)

        vis_fx = embeddings[:fake_old_feat.shape[0], 0]
        vis_fy = embeddings[:fake_old_feat.shape[0], 1]

        vis_fx = np.split(vis_fx, len(selected_old_class))
        vis_fy = np.split(vis_fy, len(selected_old_class))

        vis_x = embeddings[fake_old_feat.shape[0]:, 0]
        vis_y = embeddings[fake_old_feat.shape[0]:, 1]

        vis_x = np.split(vis_x, len(selected_old_class))
        vis_y = np.split(vis_y, len(selected_old_class))

        plt.figure(figsize=(6, 4))
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_old_class)))

        for i, (x, y) in enumerate(zip(vis_fx, vis_fy)):
            plt.scatter(x, y, s=10, marker='x', label='Synthetic', c=colors[i])

        for i, (x, y) in enumerate(zip(vis_x, vis_y)):
            plt.scatter(x, y, s=10, marker='o', label='Real', c=colors[i])

        plt.xticks([])
        plt.yticks([])
        # plt.legend()

        save_path = self.args.save_path + self.file_name + "/tsne_{}.pdf".format(self.cur_task)
        base_dir = os.path.dirname(save_path)
        if base_dir != '':
            os.makedirs(base_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        # plt.show()


def transform_invert(img, norm_transform):
    img_ = img.detach().cpu()
    mean = torch.tensor(norm_transform.mean, dtype=img_.dtype, device=img_.device)
    std = torch.tensor(norm_transform.std, dtype=img_.dtype, device=img_.device)
    img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    plt.imshow(img_)
    plt.show(block=False)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))


def normalize(tensor, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor