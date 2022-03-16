from wrap_net import WrapNet
import torch


def check_err(net, net2, x):
    o1 = net(x)
    o2 = net2(x)
    net.cuda()
    net2.cuda()
    x = x.cuda()
    o3 = net(x).cpu()
    o4 = net2(x).cpu()
    print('cpu error rate.', (o1-o2).abs().max().item())
    print('gpu error rate.', (o3-o4).abs().max().item())
    print('mix error rate.', (o1-o4).abs().max().item(), (o2-o3).abs().max().item())
    print('shape', o4.shape)


if __name__ == '__main__':
    net = WrapNet()
    jnet = torch.jit.script(net)
    torch.jit.save(jnet, 'out_jit_module.pt')

    net2 = torch.jit.load('out_jit_module.pt')
    x = torch.randn([6, 3, 320, 320])

    check_err(net, net2, x)

    print('Complete.')
