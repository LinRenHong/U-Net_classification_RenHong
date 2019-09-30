
import os
import sys
import time
import random
import datetime
import torch

from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from config import config
from utils.compiler import ModelCompiler
from models.unet_classification import UNet_classification



opt = config


class GlaucomaClassificationCompiler(ModelCompiler):

    def __init__(self, **kwargs):
        super(GlaucomaClassificationCompiler, self).__init__(**kwargs)

    def train(self):
        os.makedirs(os.path.join(self.save_images_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_models_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.results_dir, self.tb_log_path))

        # If use GPU
        if self.isCuda:
            self.model = self.model.cuda()
            self.criterion.cuda()

        # If want to retrain
        if opt.epoch != 0:
            # Load pretrained models
            print("Loading model from: [%s]" % self.load_model_path)
            self.model.load_state_dict(torch.load("%s" % self.load_model_path))
            print("Start retrain...")
        else:
            print("Start training...")

        prev_time = time.time()

        self._best_train_accuracy = 0.0

        for epoch in range(opt.epoch, opt.n_epochs):

            correct = 0
            total = 0

            for i, batch in enumerate(self.train_dataloader):
                # Model inputs
                real_A = Variable(batch["A"].type(self.tensor_type))

                ### BCE ###
                label = Variable(batch["label"].type(self.tensor_type))
                # print(label)
                one_hot_label = torch.zeros(label.size(0), opt.num_of_class).scatter_(1, label.type(
                    torch.LongTensor).unsqueeze_(-1), 1)
                one_hot_label = one_hot_label.cuda()
                # print(one_hot_label)
                ### BCE ###

                ### CrossEntropy ###
                # label = Variable(batch["label"].type(torch.cuda.LongTensor))
                ### CrossEntropy ###

                result = self.model(real_A)
                # masks_pred = result["fake_image"]
                predict_label = result["prediction"]

                _, predicted = torch.max(predict_label.data, 1)
                total += label.size(0)
                # correct += (predicted == label).sum().item()
                ### BCE ###
                correct += (predicted == label.type(torch.cuda.LongTensor)).sum().item()
                ### BCE ###

                # Calculate loss
                ### BCE ###
                self.loss = self.criterion(predict_label, one_hot_label)
                ### BCE ###
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()


                # Determine approximate time left
                batches_done = epoch * len(self.train_dataloader) + i
                batches_left = opt.n_epochs * len(self.train_dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(self.train_dataloader),
                        self.loss.item(),
                        time_left,
                    )
                )

            # Calculate accuracy
            self._train_accuracy = correct / total

            # Save model
            self.save_model(epoch=epoch)

            # write to TensorBoard
            self.write_to_tensorboard(epoch=epoch, condition="train")

            # Validate
            self.validate(epoch_done=epoch, is_save_image=True)

    def validate(self, epoch_done, is_save_image=True):
        ### Classification ###
        correct = 0
        total = 0
        ### Classification ###

        ### Generate Image ###
        temp_real_A = 0
        temp_real_B = 0
        temp_fake_B = 0

        sample_num = 9
        want_sample = random.sample(range(1, len(self.val_dataloader)), sample_num)
        ### Generate Image ###

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):

                real_A = Variable(batch["A"].type(self.tensor_type))
                real_B = Variable(batch["B"].type(self.tensor_type))
                label = Variable(batch["label"].type(torch.cuda.LongTensor))
                # print(label.shape)
                # print(real_A.shape)
                result = self.model(real_A)

                ### Generate Image ###
                if is_save_image and (i == 0 or i in want_sample):
                    if i == 0:
                        temp_real_A = real_A
                        temp_real_B = real_B
                        temp_fake_B = result["fake_image"]
                    else:
                        temp_real_A = torch.cat((temp_real_A, real_A), dim=0)
                        temp_real_B = torch.cat((temp_real_B, real_B), dim=0)
                        temp_fake_B = torch.cat((temp_fake_B, result["fake_image"]), dim=0)
                ### Generate Image ###

                predict_label = result["prediction"]
                _, predicted = torch.max(predict_label.data, 1)
                total += label.size(0)
                # correct += (predicted == label).sum().item()
                ### BCE ###
                correct += (predicted == label.type(torch.cuda.LongTensor)).sum().item()
                ### BCE ###
                print("Origin: {}".format(batch["label"]))
                print("Model Predict: {}".format(predict_label))
                print("Predict: {}\n".format(predicted))

        self._val_accuracy = correct / total
        print('Accuracy: %d %%' % (
                100 * self._val_accuracy))
        print("Correct: %s" % correct)
        print("Total: %s" % total)

        # Write to TensorBoard
        self.write_to_tensorboard(epoch=epoch_done, condition="val")

        ### Generate Image ###
        if is_save_image:
            img_sample = torch.cat((temp_real_A.data, temp_fake_B.data, temp_real_B.data), -2)
            save_image(img_sample, os.path.join(self.save_images_dir, "%s/ep(%s).png" % (self.save_ckpt_name, epoch_done)), nrow=5, normalize=True)
        ### Generate Image ###

    def test(self, is_save_image=True):

        assert self.load_model_path is not None, "Pre-train classification model is not found! Please check the pre-train classification model path"
        print("Loading model from: [%s]" % self.load_model_path)
        self.model.load_state_dict(torch.load("%s" % self.load_model_path))

        # If use GPU
        if self.isCuda:
            self.model = self.model.cuda()

        ### Classification ###
        correct = 0
        total = 0
        ### Classification ###

        ### Generate Image ###
        temp_real_A = 0
        temp_real_B = 0
        temp_fake_B = 0

        # Random sample 10 images (except first image)
        sample_num = 9
        want_sample = random.sample(range(1, len(self.val_dataloader)), sample_num)
        ### Generate Image ###

        print("Start inference...")
        # Iterate validation set
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):

                real_A = Variable(batch["A"].type(self.tensor_type))
                real_B = Variable(batch["B"].type(self.tensor_type))
                label = Variable(batch["label"].type(torch.cuda.LongTensor))
                # print(label.shape)
                # print(real_A.shape)
                result = self.model(real_A)

                ### Generate Image ###
                if is_save_image and (i == 0 or i in want_sample):
                    if i == 0:
                        temp_real_A = real_A
                        temp_real_B = real_B
                        temp_fake_B = result["fake_image"]
                    else:
                        temp_real_A = torch.cat((temp_real_A, real_A), dim=0)
                        temp_real_B = torch.cat((temp_real_B, real_B), dim=0)
                        temp_fake_B = torch.cat((temp_fake_B, result["fake_image"]), dim=0)
                ### Generate Image ###

                predict_label = result["prediction"]
                _, predicted = torch.max(predict_label.data, 1)
                total += label.size(0)
                # correct += (predicted == label).sum().item()
                ### BCE ###
                correct += (predicted == label.type(torch.cuda.LongTensor)).sum().item()
                ### BCE ###
                print("Origin: {}".format(batch["label"]))
                print("Model Predict: {}".format(predict_label))
                print("Predict: {}\n".format(predicted))
        self._val_accuracy = correct / total
        print('Accuracy: %d %%' % (
                100 * self._val_accuracy))
        print("Correct: %s" % correct)
        print("Total: %s" % total)

        ### Generate Image ###
        if is_save_image:
            img_sample = torch.cat((temp_real_A.data, temp_fake_B.data, temp_real_B.data), -2)
            save_image(img_sample, os.path.join(self.save_images_dir, "%s-test(%s)-%s-val_index(%d).png" % (self.today, self.dataset_name, self.model_name, self.val_idx)), nrow=5, normalize=True)
        ### Generate Image ###


    def save_model(self, epoch):
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            print("\nSave model to [%s] at %d epoch\n" % (self.save_ckpt_name, epoch))
            torch.save(self.model.state_dict(), os.path.join(self.save_models_dir, "%s/%s_%d.pth" % (self.save_ckpt_name, self.model_name, epoch)))

        # Save best model
        if self._train_accuracy > self._best_train_accuracy:
            self._best_train_accuracy = self._train_accuracy
            print("\nSave best model to [%s]\n" % self.save_ckpt_name)
            torch.save(self.model.state_dict(), os.path.join(self.save_models_dir, "%s/best_%s_%d.pth" % (self.save_ckpt_name, self.model_name, epoch)))

        # Save latest model
        if epoch == (opt.n_epochs - 1):
            print("\nSave latest model to [%s]\n" % self.save_ckpt_name)
            torch.save(self.model.state_dict(), os.path.join(self.save_models_dir, "%s/%s_%d.pth" % (self.save_ckpt_name, self.model_name, opt.n_epochs)))



    def write_to_tensorboard(self, epoch, condition):
        if self.writer is not None:

            if condition.strip() in ["train", "Train", "TRAIN"]:
                self.writer.add_scalar(tag='Loss', scalar_value=self.loss.item(), global_step=epoch)
                self.writer.add_scalar(tag='Accuracy_train', scalar_value=self._train_accuracy, global_step=epoch)

            elif condition.strip() in ["val", "Val", "VAL"]:
                self.writer.add_scalar(tag='Accuracy_val', scalar_value=self._val_accuracy, global_step=epoch)

            else:
                print("Please specify condition: [\"train\", \"Train\", \"TRAIN\"] or [\"val\", \"Val\", \"VAL\"]")
        else:
            print("Writer is None!")



if __name__ == '__main__':
    opt = config
    model = UNet_classification(3, 3)
    # print(model)
    x = torch.randn(1, 3, opt.img_crop_height, opt.img_crop_width)
    result = model(x)
    print("Output Image: {}".format(result["fake_image"]))
    print("Prediction: {}".format(result["prediction"]))
