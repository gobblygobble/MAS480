# for code implementation
import torch
import numpy as np
import torch.nn as nn
#from google.colab import drive
from torch.optim import SGD, Adam
from torch.autograd import Variable
# for plotting graphs & data loader
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid
## IMPORT MY LIBRARIES
# for early stopping with metrics
# network definitions
import unet
import metrics
# others
from argparse import ArgumentParser
from dataset import VOC12
from criterion import CrossEntropyLoss2d
from transform import Relabel, ToLabel, Colorize, colormap
# variables - fixed for VOC2012 data set
NUM_CHANNELS = 3
NUM_CLASSES = 22
# image transforms
color_transform = Colorize()
image_transform = ToPILImage()
input_transform = Compose([
    CenterCrop(256), # 256x256 cropped image
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]), # mean, std for each channels
])
target_transform = Compose([
    CenterCrop(256), # 256x256 cropped image
    ToLabel(),
    Relabel(255, 21),
])
'''
target_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize((0),(1/256))
])
'''
cmap = colormap(NUM_CLASSES)[:, np.newaxis, :]

## training
def train(args, model):
    # set model to training mode
    model.train()
    # prepare criterion
    weight = torch.ones(22)
    weight[0] = 0
    train_loader = DataLoader(
                            VOC12(root=args.datadir, train=True, input_transform=input_transform,
                                  target_transform=target_transform),
                            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(
                            VOC12(root=args.datadir, train=False,input_transform=input_transform,
                                  target_transform=target_transform),
                            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    # use Adam optimizer
    optimizer = Adam(model.parameters())
    # for loss calculation, we still use CrossEntropyLoss
    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)
    # start training - epoch values start from 1 to make numbers look 'pretty'
    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = []
        for step, (images, labels) in enumerate(train_loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)
            # refresh gradient before backprop
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            if (args.steps_loss > 0) and (step % args.steps_loss == 0):
                # print loss
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average} (epoch {epoch}, step {step})')

                # check for early stop
                if metrics.early_stop(use_cuda=args.cuda, output=outputs, target=targets[:, 0], metric="iou", threshold=0.8):
                    # if we can stop early, save model and exit
                    print("Early stopping score exceeded threshold... saving model and ending training stage")
                    if args.attention:
                        filename = f'models/AttentionUNet-EarlyStop-{epoch:03}-{step:04}.pth'
                    else:
                        filename = f'models/UNet-EarlyStop-{epoch:03}-{step:04}.pth'
                    torch.save(model.state_dict(), filename)
                    return
            if (args.steps_save > 0) and (step % args.steps_save == 0):
                if args.attention:
                    filename = f'models/AttentionUNet-{epoch:03}-{step:04}.pth'
                else:
                    filename = f'models/UNet-{epoch:03}-{step:04}.pth'
                torch.save(model.state_dict(), filename)
                print(f'save: {filename} (epoch: {epoch}, step: {step})')
                ''' printing out pictures doesn't work on VScode :(
                _, outputs = torch.max(outputs, dim=1)
                #print(outputs.shape)
                outputs = outputs.unsqueeze(1)
                #print(outputs.shape)
                # show images
                plt.subplot(211)
                plt.imshow(make_grid(images.cpu()).permute(1,2,0).numpy())
                plt.axis('off')
                plt.title('Images')
                # print labels
                plt.subplot(212)
                outputs = outputs.cpu().numpy()[:, :, :, :, np.newaxis]
                color_Label = np.dot(outputs == 0, cmap[0])
                for i in range(1, cmap.shape[0]):
                    color_Label += np.dot(outputs == i, cmap[i])
                color_Label = color_Label.swapaxes(1,4)
                plt.imshow((make_grid(torch.tensor(color_Label.squeeze())).permute(1,2,0).numpy()).astype('uint8'))
                plt.axis('off')
                plt.title('Label')
                '''
        # every epoch, check validation data's accuracy
        v_loss_list = []
        for _, (v_images, v_labels) in enumerate(val_loader):
            if args.cuda:
                v_images = v_images.cuda()
                v_labels = v_labels.cuda()
            v_inputs = Variable(v_images)
            v_targets = Variable(v_labels)
            v_outputs = model(v_inputs)
            v_loss = criterion(v_outputs, v_targets[:, 0])
            v_loss_list.append(v_loss.item())
        v_average = sum(v_loss_list) / len(v_loss_list)
        print(f'validation loss: {v_average} (epoch {epoch})')

## evaluation
def evaluate(args, model):
    # set model to evaluation mode
    model.eval()

    image = input_transform(Image.open(args.image))
    label = model(Variable(image, volatile=True).unsqueeze(0))
    label = color_transform(label[0].data.max(0)[1])

    image_transform(label).save(args.label)

def main(args):
    # load correct model
    if args.attention == True:
        model = unet.AttentionUnet(in_channels=NUM_CHANNELS,out_channels=NUM_CLASSES)
    elif args.attention == False:
        # we don't use residual for non-denoising purposes
        model = unet.Unet(in_channels=NUM_CHANNELS,out_channels=NUM_CLASSES)
    # check if we use GPU
    if args.cuda:
        model = model.cuda()
    # check for mode - training or evaluation?
    if args.mode == 'eval':
        evaluate(args, model)
    elif args.mode == 'train':
        train(args, model)

## actual running
if __name__ == '__main__':
    # comment out lines from original code that is not used here
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--attention', action='store_true')
    #parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('image')
    parser_eval.add_argument('label')

    parser_train = subparsers.add_parser('train')
    #parser_train.add_argument('--port', type=int, default=80)
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--batch-size', type=int, default=4)
    parser_train.add_argument('--steps-loss', type=int, default=250)
    #parser_train.add_argument('--steps-plot', type=int, default=0)
    parser_train.add_argument('--steps-save', type=int, default=500)
    ''' This part is only for Jupyter Python Notebook
    # since we can't pass CLI arugments in Python notebook,
    # assign variables manually
    # add two spaces between each chunk so that the data directory doesn't get split
    #command_line = "--cuda  train  --datadir  gdrive/My Drive/Project/data  --num-epochs  1  --num-workers  4  --batch-size  4  --steps-save  500"
    #command_line = "--cuda  train  --datadir  data  --num-epochs  1  --num-workers  4  --batch-size  4  --steps-save  500"
    #command_line = "train  --datadir  data  --num-epochs  1  --num-workers  4  --batch-size  4  --steps-save  500"
    args = parser.parse_args(command_line.split("  "))
    print(args)
    '''
    # an example would be python main.py --cuda --attention train --datadir data --num-epochs 320
    args = parser.parse_args()
    main(args)