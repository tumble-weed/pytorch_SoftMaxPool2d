import torch

class SoftMaxPool2d(torch.nn.Module):
    def __init__(self,T,wsize,wstep):
        assert len(wsize) == 2
        super(SoftMaxPooling,self).__init__()
        self.T = torch.tensor(T).float()
        
        self.wsize = wsize
        self.wstep = wstep 

    def forward(self,t):
        assert t.dim() == 4
        u = t.unfold(2,self.wsize[0],self.wstep).unfold(3,self.wsize[1],self.wstep)
        w = u.contiguous().view(u.shape[:4] + (-1,))
        w_by_T = w/self.T
        softmaxed = torch.nn.functional.softmax(w_by_T,-1)
        pooled = torch.sum(softmaxed*w,-1)
        return pooled
