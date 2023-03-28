import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict
import torch.utils.data as data_utils
import torch

class TeacherStudentTrainer:
    def __init__(self):
        self.opt = TrainOptions().parse()
        self.data_loader = CreateDataLoader(self.opt)
        self.dataset = self.data_loader.load_data()
        self.dataset_size = len(self.data_loader)
        print('#training images = %d' % self.dataset_size)

        self.teacher_model = create_model(self.opt)
        self.teacher_model.setup(self.opt)

        self.student_model = create_model(self.opt)
        self.student_model.setup(self.opt)

        self.visualizer = Visualizer(self.opt)
        self.total_steps = 0
        self.total_steps = 0
        self.iter_data_time = time.time()

    def update_teacher_model(self, keep_rate=0.9996):
        student_model_dict = self.student_model.state_dict()
        teacher_model_dict = self.teacher_model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in teacher_model_dict.items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.teacher_model.load_state_dict(new_teacher_dict)


    def train(self):
        for epoch in range(self.opt.epoch_count, self.opt.niter + self.opt.niter_decay + 1):
            self.epoch = epoch
            self.epoch_start_time = time.time()
            self.iter_data_time = time.time()
            if epoch < self.opt.teacher_train_iter:
                self.train_teacher_loop()
            else:

                if epoch == self.opt.teacher_train_iter:
                    # evaluate unlabeled dataset with teacher model and obtain uncertainty scores
                    self.uncertainty_scores = []
                    for i, data in enumerate(self.dataset):
                        self.teacher_model.set_input(data)
                        self.teacher_model.forward()
                        uncertainty = self.teacher_model.get_uncertainty()
                        self.uncertainty_scores.append((i, uncertainty))
                    self.uncertainty_scores.sort(key=lambda x: x[1])

                self.teacher_model.save_networks('teacher') # Save teacher model
                self.student_model = torch.load('teacher.pt') # Get student model state dict

                # change student to copy teacher at this point    
                if self.epoch == self.opt.SEMISUPNET.BURN_UP_STEP:
                    # update copy the the whole model
                    self.update_teacher_model(keep_rate=0.00)

                elif (
                    self.epoch - self.opt.SEMISUPNET.BURN_UP_STEP
                ) % self.opt.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                    self.update_teacher_model(
                        keep_rate=self.opt.SEMISUPNET.EMA_KEEP_RATE)
                # gradually add data with lower uncertainty scores
                threshold = self.opt.threshold
                high_confidence_indices = [i for i, score in self.uncertainty_scores if score < threshold]
                high_confidence_dataset = torch.utils.data.Subset(self.dataset, high_confidence_indices)
                self.high_confidence_loader = data_utils.DataLoader(high_confidence_dataset, batch_size=self.opt.batchSize, shuffle=True)
                
                # train student model on high confidence data
                self.train_student_loop(high_confidence_loader)
                
                # gradually lower the threshold to include more data with lower confidence scores
                threshold -= self.opt.threshold_step
                threshold = max(threshold, self.opt.min_threshold)
    
    def train_teacher_loop(self):   
        self.epoch_iter = 0
        for i, data in enumerate(self.dataset):
            iter_start_time = time.time()
            if self.total_steps % self.opt.print_freq == 0:
                t_data = iter_start_time - self.iter_data_time
            self.visualizer.reset()
            self.total_steps += self.opt.batchSize
            self.epoch_iter += self.opt.batchSize
            self.teacher_model.set_input(data)
            self.teacher_model.optimize_parameters()

            if self.total_steps % self.opt.display_freq == 0:
                save_result = self.total_steps % self.opt.update_html_freq == 0
                self.visualizer.display_current_results(self.teacher_model.get_current_visuals(), epoch, save_result)

            if self.total_steps % self.opt.print_freq == 0:
                losses = self.teacher_model.get_current_losses()
                t = (time.time() - iter_start_time) / self.opt.batchSize
                self.visualizer.print_current_losses(self.epoch, self.epoch_iter, losses, t, t_data)
                if self.opt.display_id > 0:
                    self.visualizer.plot_current_losses(self.epoch, float(self.epoch_iter) / self.dataset_size, self.opt, losses)

            if self.total_steps % self.opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (self.epoch, self.total_steps))
                self.teacher_model.save_networks('latest')

            self.iter_data_time = time.time()

    def train_student_loop(self):   
        self.epoch_iter = 0
        for i, data in enumerate(self.high_confidence_loader):
            iter_start_time = time.time()
            if self.total_steps % self.opt.print_freq == 0:
                t_data = iter_start_time - self.iter_data_time
            self.visualizer.reset()
            self.total_steps += self.opt.batchSize
            self.epoch_iter += self.opt.batchSize
            self.teacher_model.set_input(data)
            self.teacher_model.optimize_parameters()

            if self.total_steps % self.opt.display_freq == 0:
                save_result = self.total_steps % self.opt.update_html_freq == 0
                self.visualizer.display_current_results(self.teacher_model.get_current_visuals(), epoch, save_result)

            if self.total_steps % self.opt.print_freq == 0:
                losses = self.teacher_model.get_current_losses()
                t = (time.time() - iter_start_time) / self.opt.batchSize
                self.visualizer.print_current_losses(self.epoch, self.epoch_iter, losses, t, t_data)
                if self.opt.display_id > 0:
                    self.visualizer.plot_current_losses(self.epoch, float(self.epoch_iter) / self.dataset_size, self.opt, losses)

            if self.total_steps % self.opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                        (self.epoch, self.total_steps))
                self.teacher_model.save_networks('latest')

            self.iter_data_time = time.time()
