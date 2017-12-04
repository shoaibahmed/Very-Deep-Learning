from optparse import OptionParser
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# PyTorch imports
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision

import os
import pandas as pd
# from skimage import io, transform
from PIL import Image

# Command line options
parser = OptionParser()

parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=224, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=224, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")
parser.add_option("--usePretrainedModel", action="store_true", dest="usePretrainedModel", default=False, help="Whether to use pretrained model or start training from scratch")

parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=20, help="Batch size")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Training epochs")
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-3, help="Learning Rate")
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=120, help="Number of classes")

parser.add_option("--numTrainingInstances", action="store", type="int", dest="numTrainingInstances", default=60000, help="Training instances")
parser.add_option("--numTestInstances", action="store", type="int", dest="numTestInstances", default=10000, help="Test instances")

# Parse command line options
(options, args) = parser.parse_args()
options.cuda = torch.cuda.is_available()
print (options)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class DogBreedsDataset(torch.utils.data.Dataset):
    """Dog breeds dataset."""

    def __init__(self, csv_file, root_dir, transform=None, loader=default_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file, sep=',', header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

        self.classNames = np.unique(self.labels_frame.ix[:, 1])
        self.clsIdx = dict(zip(self.classNames, range(len(self.classNames))))

        # print (self.indCls)
        # print (len(self.indCls))
        print (self.clsIdx)

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.ix[idx, 0] + '.jpg')
        # image = io.imread(img_name)
        image = self.loader(img_name)
        label = self.labels_frame.ix[idx, 1]
        labelIdx = self.clsIdx[label]
        # sample = {'image': image, 'label': labelIdx, 'label_name': label}

        if self.transform:
            image = self.transform(image)

        return image, labelIdx, label

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
dogs_dataset = DogBreedsDataset(csv_file='./data/labels.csv', root_dir='./data/train/', transform=data_transform)
dataloader = torch.utils.data.DataLoader(dogs_dataset, batch_size=options.batchSize, shuffle=True, num_workers=4)

'''
# Test to check dataset
for i in range(len(dogs_dataset)):
    fig = plt.figure()
    im, labelIdx, clsName = dogs_dataset[i]
    print(i, labelIdx, clsName, im.size())
    ax = plt.subplot()
    plt.tight_layout()
    ax.set_title('Sample # {} | Class: {} | Class Idx: {}'.format(i, clsName, labelIdx))
    ax.axis('off')

    plt.imshow(im.numpy().transpose([1, 2, 0]))
    # plt.imshow(im)
    plt.show()
    plt.close()
'''

model = torchvision.models.resnet152(pretrained=options.usePretrainedModel)
if options.cuda:
    model.cuda()

# Define the loss
loss = torch.nn.CrossEntropyLoss()

# Define the optimizer
optim = torch.optim.Adam(model.parameters(), lr=options.learningRate)

def testModel():
    # Perform test step
    model.eval()
    lossTest = 0.0
    correct = 0
    for data, target in test_loader:
        if options.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data, volatile=True), torch.autograd.Variable(target)
        output = model(data)
        # lossTest += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        lossTest += loss(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return lossTest

for epoch in range(options.trainingEpochs):
    # Perform train step
    step = 0
    for batch_idx, (data, target, targetClsName) in enumerate(dataloader):
        # Compute test loss first since the error reported after optimization will be lower than the train error
        # lossTest = testModel()
        # print (len(data))
        # print ("Data shape: %s | Target shape: %s | Target cls name shape: %s" % (str(data.size()), str(target.size()), str(targetClsName.size())))

        # Perform the training step
        model.train()
        if options.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        optim.zero_grad()
        output = model(data)
        currentLoss = loss(output, target)
        lossTrain = currentLoss.data[0]
        currentLoss.backward()
        optim.step()

        # print ("Epoch: %d | Step: %d | Train Loss: %f | Test Loss: %f" % (epoch, step, lossTrain, lossTest))
        print ("Epoch: %d | Step: %d | Train Loss: %f" % (epoch, step, lossTrain))
        step += 1
