import torch

class SoftMaxPool2d(torch.nn.Module):
    def __init__(self,T,wsize,wstep,padding=0):
        assert len(wsize) == 2
        super(SoftMaxPool2d,self).__init__()
        self.T = torch.tensor(T).float()
        
        self.wsize = wsize
        self.wstep = wstep 
        self.padding = padding
    def forward(self,t):
        if self.padding > 0:
            padder = torch.nn.ZeroPad2d(self.padding)
            t = padder(t)
        assert t.dim() == 4
        u = t.unfold(2,self.wsize[0],self.wstep).unfold(3,self.wsize[1],self.wstep)
        w = u.contiguous().view(u.shape[:4] + (-1,))
        w_by_T = w/self.T
        softmaxed = torch.nn.functional.softmax(w_by_T,-1)
        pooled = torch.sum(softmaxed*w,-1)
        return pooled
