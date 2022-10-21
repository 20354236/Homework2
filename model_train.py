import os

import model_define
from torch import nn
from model_functions import *
my_Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net=model_define.TinySSD(num_classes=1)
net=net.to(my_Device)
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)


###    加载数据
batch_size = 32
train_iter = load_data(batch_size)

if not os.path.exists('./pkl'):
    os.makedirs('./pkl')
num_epochs = 81     #20
for epoch in range(num_epochs):
    print('epoch: ', epoch)
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = Accumulator(4)
    net.train()
    for features, target in train_iter:
        trainer.zero_grad()
        X, Y = features.to(my_Device), target.to(my_Device)            #数据和目标
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)       #anchors为对应所有图的框的 左上右下坐标
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks,cls_loss,bbox_loss)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                    bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric.data[0] / metric.data[1], metric.data[2] / metric.data[3]
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')

    #保存模型参数
    if epoch % 10 == 0:
        torch.save(net.state_dict(), './pkl/'+'net_' + str(epoch) + '.pkl')