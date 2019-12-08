import h5py
import math
import os
import numpy as np
import torch
import torchlight
import torch.optim as optim
import torch.nn as nn
from net import classifier
import pickle



class MAPELoss(nn.Module):
    """
    Reverse Hubber Loss function
    """
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predicted_out, target):
        """
        Function computes the loss in the forward pass between the target and the predicted output
        :param predicted_out:
        :param target:
        :return:
        """
        ret = torch.mean(torch.abs((target - predicted_out)/target))
        #print("Return", ret)
        return ret


class ReverseHubberLoss(nn.Module):
    """
    Reverse Hubber Loss function
    """
    def __init__(self):
        super(ReverseHubberLoss, self).__init__()
        self.c = 0

    def forward(self, predicted_out, target):
        """
        Function computes the loss in the forward pass between the target and the predicted output
        :param predicted_out:
        :param target:
        :return:
        """
        ret = self._reverse_hubber_loss(predicted_out, target)
        #print("Return", ret)
        return torch.mean(ret)

    def _reverse_hubber_loss(self, a_input, a_target):
        """
        Function computes the reverse hubber loss between the input and the target labels
        :param input:
        :param target:
        :return:
        """
        #if a_target.requires_grad:
        error = torch.abs(a_input - a_target)

        #print("Error", error)
        self.c = 0.2*torch.max(error)
        #print(self.c)
        loss = torch.where(error <= self.c, torch.abs(error), (error**2 + self.c**2)/(2*self.c))
        #print("Loss", loss)
        return loss

# def get_best_epoch_and_accuracy(path_to_model_files):
#     all_models = os.listdir(path_to_model_files)
#     while '_' not in all_models[-1]:
#         all_models = all_models[:-1]
#     best_model = all_models[-1]
#     all_us = list(find_all_substr(best_model, '_'))
#     return int(best_model[5:all_us[0]]), float(best_model[all_us[0]+4:all_us[1]])


class Processor(object):
    """
        Processor for performing the depth prediction. This class handles helper functions such as training, testing
        and evaluation, saving of logs, printing logs onto console.
        Invokes the Depth Prediction class to build the model, Dataset loader class to load the data
    """

    def __init__(self, args, data_loader, device='cuda:0'):

        self.args = args
        # TBD:The data loader class has to be implemented to load the train, test and evaluation images
        self.data_loader = data_loader
        #self.kf = KFold(n_splits=2)
        self.result = dict()
        self.iter_info = dict()
        self.train_epoch_info = dict()
        self.eval_epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.device = device
        self.io = torchlight.IO(
            "./model_output/",
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # Initialize timer to see the time taken for each batch
        self.io.init_timer("batch_processing_time")
#
        # model
        self.model = classifier.DepthPredictionNet()
        self.model.cuda('cuda:0')
        # TBD: Apart from the Resnet layers, all other layers has to be initialized with the weights as mentioned in the
        # paper, For now we can go ahead and try the default initialization which can be later modified
        # [Resolved: Doing at Model level]
        self.loss = ReverseHubberLoss()
        self.mean_loss = nn.MSELoss()
        self.abs_loss = MAPELoss()
        self.best_loss = math.inf

        self.train_loss_save = dict()
        self.cross_loss_save = dict()
        #self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = None
        self.best_mean_error = np.zeros((1, np.max(self.args.topk)))
        self.mean_error_updated = False


        self.train_loss_save['mean_loss'] = []
        self.train_loss_save['rmse_loss'] = []
        self.train_loss_save['abs_loss'] = []

        self.cross_loss_save['mean_loss'] = []
        self.cross_loss_save['abs_loss'] = []
        self.cross_loss_save['mean_loss'] = []


        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr
        self.mean_loss_per_lr_step = 0

    def adjust_lr(self):

        if self.args.optimizer == 'SGD' and (self.meta_info['epoch'] % self.args.step == 0):
            if np.fabs(self.mean_loss_per_lr_step - np.mean(self.eval_epoch_info['mean_loss'])) < 0.001:
                lr = self.lr * 0.5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            self.mean_loss_per_lr_step = np.mean(self.eval_epoch_info['mean_loss'])
#
    def show_epoch_info(self, data_item):

        for k, v in data_item.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        # TBD: Ignore the pavi log for now. Will handle that later
        # [Resolved: This is not being used in the parent system]
        # if self.args.pavi_log:
        #     self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

        # TBD: Ignore the pavi log for now. Will handle that later[Resolved - Not being used]
#             if self.args.pavi_log:
#                 self.io.log('train', self.meta_info['iter'], self.iter_info)

    # def show_topk(self, k):
    #     """
    #     The function determines the loss of the latest model
    #     :param k:
    #     :return:
    #     """
    #     # rank = self.result.argsort()
    #     # hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
    #     # accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
    #     # if accuracy > self.best_accuracy[0, k-1]:
    #     #     self.best_accuracy[0, k-1] = accuracy
    #     #     self.accuracy_updated = True
    #     # else:
    #     #     self.accuracy_updated = False
    #     self.io.print_log('\tTop{}: {:.2f}%. Best so far: {:.2f}%.'.format(k, accuracy, self.best_accuracy[0, k-1]))
#
    def per_train(self, train_data):
        loss_value = []
        rmse_loss_value = []
        abs_loss_value = []

        for data, label in train_data:
            # get data
            data = data.float().to(self.device)
            label = label.float().to(self.device)

            # forward pass and compute the loss
            output = self.model(data)
            loss = self.loss(output, label)
            rmse_loss = torch.sqrt(self.mean_loss(output, label))
            abs_loss = self.abs_loss(output, label)

            # backward and update the weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            rmse_loss_value.append(rmse_loss.item())
            abs_loss_value.append(abs_loss.item())
            self.train_loss_save['mean_loss'].append(self.iter_info['loss'])
            self.train_loss_save['rmse_loss'].append(rmse_loss.item())
            self.train_loss_save['abs_loss'].append(abs_loss.item())
            self.show_iter_info()
            self.meta_info['iter'] += 1

        return np.mean(loss_value), np.sqrt(np.mean(np.square(rmse_loss_value))), np.mean(abs_loss_value)

        # TBD: Ignore the topk block of code for now
#         # for k in self.args.topk:
#         #     self.calculate_topk(k, show=False)
#         # if self.accuracy_updated:
#         # self.model.extract_feature()

    def per_test(self, test_data, evaluation=True):

        self.model.eval()
        # TBD: Once the loader class has been implemented then the dataset can be clearly demarcated
        # [Resolved]: The loader class has been implemented for now.
        loss_value = []
        rmse_loss_value = []
        abs_loss_value = []
        result_frag = []
        label_frag = []

        for data, label in test_data:

            # get data from the loader class
            data = data.float().to(self.device)
            label = label.float().to(self.device)

            # Forward pass to obtain the output
            with torch.no_grad():
                output = self.model(data)
            # Append the output of the prediction map
            #result_frag.append(output.data.cpu().numpy())

            # get loss in case of evaluation flag set to True
            if evaluation:
                loss = self.loss(output, label)
                rmse_loss = torch.sqrt(self.mean_loss(output,label))
                abs_loss = self.abs_loss(output, label)

                loss_value.append(loss.item())
                rmse_loss_value.append(rmse_loss.item())
                abs_loss_value.append(abs_loss.item())
                label_frag.append(label.data.cpu().numpy())

                self.cross_loss_save['mean_loss'].append(loss.item())
                self.cross_loss_save['abs_loss'].append(rmse_loss.item())
                self.cross_loss_save['mean_loss'].append(abs_loss.item())
        # Store the depth prediction maps for each image in dictionary
        # self.result = np.concatenate(result_frag)
        if evaluation:
            # Store the depth label data for each image in dictionary
            # self.label = np.concatenate(label_frag)
            loss_value = np.mean(loss_value)
            rmse_loss_value = np.sqrt(np.mean(np.square(rmse_loss_value)))
            abs_loss_value = np.mean(abs_loss_value)
        return (loss_value, rmse_loss_value, abs_loss_value)

#        TBD: Ignore the show top-k accuracy which will be resolved later. Keep ths aside for now. Anyways we are
#        concerned about this when we really need the top k factor mean loss. Which is not our concern now.
#             # show top-k accuracy
#             for k in self.args.topk:
#                 self.show_topk(k)
#
    def train(self):
        # TBD: Implement the class to load the dataset.
        # [Resolved] : Class implemented
        train_loader = self.data_loader['train']
        eval_loader = self.data_loader['validation']

        kf_train_loss_value = []

        berhu_loss_value = []
        rmse_loss_value = []
        abs_loss_value = []

        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.model.train()
            self.meta_info['epoch'] = epoch

            # Reset time to current
            self.io.record_time()

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))

            # Running the KFold cross validation and averaging the error value for each epoch
            kf_train_loss_value = self.per_train(train_loader)

            kf_test_loss_value = self.per_test(eval_loader)
            berhu_loss_value = kf_test_loss_value[0]
            rmse_loss_value = kf_test_loss_value[1]
            abs_loss_value = kf_test_loss_value[2]

            # Showing the mean loss for the training epoch
            self.train_epoch_info['mean_loss'] = kf_train_loss_value[0]
            self.train_epoch_info['rmse_loss'] = kf_train_loss_value[1]
            self.train_epoch_info['abs_rel_loss'] = kf_train_loss_value[2]
            self.show_epoch_info(self.train_epoch_info)
            self.io.print_log('Done.')

            # evaluation showing all the losses for the cross validation
            self.io.print_log('Cross validation Eval epoch: {}'.format(epoch))
            self.eval_epoch_info['mean_loss'] = berhu_loss_value
            self.eval_epoch_info['rmse_loss'] = rmse_loss_value
            self.eval_epoch_info['abs_rel_loss'] = abs_loss_value
            self.show_epoch_info(self.eval_epoch_info)
            self.io.print_log('Done.')

            # print Epoch processing time
            self.io.check_time("batch_processing_time")
            self.io.print_timer()

            self.adjust_lr()
            # TBD: save model and weights.
            # [Resolved] Only needs to be verified
            if self.best_loss - self.eval_epoch_info['mean_loss'] > 0.001:
                self.best_loss = self.eval_epoch_info['mean_loss']
                self.best_epoch = epoch
                torch.save(self.model.state_dict(),
                           os.path.join('./model_output/',
                                        'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, 1)))
        pickle.dump(self.train_loss_save, open('../data/train_dump.txt', 'wb'))


        pickle.dump(self.cross_loss_save, open('../data/cross_dump.txt', 'wb'))

    # The function is not being used. So lets not care about this for now.
    def test(self):

        # the path of weights must be appointed
        # if self.args.weights is None:
        #     raise ValueError('Please appoint --weights.')
        # self.io.print_log('Model:   {}.'.format(self.args.model))
        # self.io.print_log('Weights: {}.'.format(self.args.weights))
        # evaluation
        loader = self.data_loader['test']
        self.model.eval()

        self.io.print_log('Test Evaluation Start:')
        test_loss = self.per_test(loader)
        self.eval_epoch_info['mean_loss'] = test_loss[0]
        self.eval_epoch_info['rmse_loss'] = test_loss[1]
        self.eval_epoch_info['abs_rel_loss'] = test_loss[2]
        self.show_epoch_info(self.eval_epoch_info)
        self.io.print_log('Done.\n')

        # save the output of model
        # if self.args.save_result:
        #     result_dict = dict(
        #         # TBD: Data loader class once implemented will be able to save the weights automatically
        #         zip(self.data_loader['test'].dataset.sample_name,
        #             self.result))
        #     self.io.save_pkl(result_dict, 'test_result.pkl')

    # TBD: The saving of best features is to be done later after performing a dry run.
    # def save_best_feature(self, ftype, data, joints, coords):
    #     if self.best_epoch is None:
    #         self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(self.args.work_dir)
    #     else:
    #         best_accuracy = self.best_accuracy.item()
    #     filename = os.path.join(self.args.work_dir,
    #                             'epoch{}_acc{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
    #     print(filename)
    #     self.model.load_state_dict(torch.load(filename))
    #     features = np.empty((0, 256))
    #     fCombined = h5py.File('../data/features2D'+ftype+'.h5', 'r')
    #     fkeys = fCombined.keys()
    #     dfCombined = h5py.File('../data/deepFeatures'+ftype+'.h5', 'w')
    #     for i, (each_data, each_key) in enumerate(zip(data, fkeys)):
    #
    #         # get data
    #         each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
    #         each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
    #         each_data = torch.from_numpy(each_data).float().to(self.device)
    #
    #         # get feature
    #         with torch.no_grad():
    #             _, feature = self.model(each_data)
#                 fname = [each_key][0]
#                 dfCombined.create_dataset(fname, data=feature.cpu())
#                 features = np.append(features, np.array(feature.cpu()).reshape((1, feature.shape[0])), axis=0)
#         dfCombined.close()
#         return features
