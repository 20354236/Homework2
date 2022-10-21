import model_define
import glob
import torchvision
from model_visualization import *
net=model_define.TinySSD(num_classes=1)
net.load_state_dict(torch.load('./pkl/net_40.pkl'))

files = glob.glob('detection/test/*.jpg')
for name in files:
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
    output = predict(X,net)
    display(img, output.cpu(), threshold=0.8)

