import json
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plot
from torchvision.transforms import transforms
from Model.net import CNN
from dataset.datates import CaptchaData
from utils.tools import img_loader

#加载配置文件
with open('./conf/sample_config.json', encoding='utf-8') as f:
    sample_json = json.loads(f.read())
model_path =  sample_json['model_save_path']



def predict(img_dir='./data/test'):
    """

    :param img_dir: 文件路径
    :return:
    """
    transforms = Compose([ToTensor()])
    alphabet = sample_json['char_set']
    dataset = CaptchaData(img_dir,
                          transform=transforms,
                          sample_conf=sample_json)
    cnn = CNN(sample_json)
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))
    acc_=[]
    for k, (img, target) in enumerate(dataset):
        img.unsqueeze_(0)
        img = img.cuda()
        target = target.view(1, 4*36).cuda()
        output = cnn(img)
        output = output.view(-1, 36)
        target = target.view(-1, 36)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output = output.view(-1, 4)[0]
        target = target.view(-1, 4)[0]
        pred=''.join([alphabet[i] for i in output.cpu().numpy()])
        print('pred: '+pred)
        true=''.join([alphabet[i] for i in target.cpu().numpy()])
        print('true: '+true )
        plot.imshow(img.permute((0, 2, 3, 1))[0].cpu().numpy())
        plot.show()
        if pred==true:
            acc_.append(1)
        else:
            acc_.append(0)
    print(sum(acc_)/len(acc_))


        # if k >= 10: break


def predict_local_one(path):

    char_set=sample_json['char_set']
    max_captcha=sample_json['max_captcha']
    cnn = CNN(sample_json)
    cnn.load_state_dict(torch.load(model_path))
    ToTnesor = transforms.ToTensor()
    img =ToTnesor(img_loader(path))
    img.unsqueeze_(0)
    output=cnn(img)
    output = output.view(-1, len(char_set)*int(max_captcha))
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    output = output.view(-1,1)[0]
    alphabet=sample_json ['char_set']
    text=''.join([alphabet[i] for i in output.detach().numpy()])
    return text



if __name__=="__main__":
    text=predict_local_one(r'D:\Py\基于pytorch的中文字体识别\dataset\1559295785540代_2.jpg')
    print(text)
    # predict()