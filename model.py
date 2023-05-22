import torch 
import torch.nn as nn

class FE(nn.Module):
    def __init__(self, channel_size=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,30,kernel_size=(1,25)),
            nn.Conv2d(30,30, kernel_size=(channel_size, 1)),
            nn.AvgPool2d(kernel_size=(1, 75), stride=15),
            nn.Flatten(),
            nn.Linear(15960, 64)
        )
        
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.dis(x)

class Classifier(nn.Module) :
    def __init__(self, input_size=64, cls_num=3) :
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, cls_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x) :
        return self.fc(x)

def Wasserstein_Loss(dc_s, dc_t) :
    return torch.mean(dc_s) - torch.mean(dc_t)

class CenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            
    #num_classes (int): number of classes.
    #feat_dim (int): feature dimension.
    def forward(self, x, labels):
        # x: feature matrix with shape (batch_size, feat_dim).
        # labels: ground truth labels with shape (batch_size).
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = torch.addmm(distmat, x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_classes).long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
if __name__ == '__main__' :
    
    fe = FE().cuda()
    dis = Discriminator().cuda()
    optim_dis = torch.optim.Adam(dis.parameters(), lr=0.0005)
    src = torch.randn(64,1,32,8064).cuda()
    trg = torch.randn(64,1,32,8064).cuda()
    optim_dis.zero_grad()
    for p in dis.parameters() :
        p.requires_grad = True
    feat_s = fe(src)
    feat_t = fe(trg)
    dc_s = dis(feat_s)
    dc_t = dis(feat_t)
    epsil = torch.rand(1).item()
    feat = epsil*feat_s + (1-epsil)*feat_t
    dc = dis(feat)
    grad = torch.autograd.grad(outputs=dc, inputs=feat, grad_outputs=torch.ones(dc.size()).cuda(), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.sqrt(torch.sum(grad**2, dim=1)+1e-12)
    grad_loss = ((grad_norm-1)**2).mean()
    
    print(grad_loss)
    print(Wasserstein_Loss(dc_s, dc_t))
    print(-Wasserstein_Loss(dc_s, dc_t)+10*grad_loss)
    loss = -Wasserstein_Loss(dc_s, dc_t)+10*grad_loss
    loss.backward()
    optim_dis.step()
    