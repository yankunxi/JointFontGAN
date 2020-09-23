import re
import matplotlib.pyplot as plt

# fp = open(
#     '/mnt/Files/XIremote/OneDrive - Wayne State University/XIcodes/Python/pytorch/pytorch1.0/MC-GAN_azadis/MC-GAN_1/checkpoints_64_1000/GlyphNet_pretrain/output.txt')

fps = []
fps += [open('/mnt/Files/XIremote/OneDrive - Wayne State '
             'University/XIcodes/Python/python3.6/pytorch1.2'
             '/XIfontGAN_yankunxi/XIFONTGAN_05/xifontgan'
             '/checkpoints/Capitals64_EskGAN2_dspostG=1/_log_train.txt')]
fps += [open('/mnt/Files/XIremote/OneDrive - Wayne State '
             'University/XIcodes/Python/python3.6/pytorch1.2'
             '/XIfontGAN_yankunxi/XIFONTGAN_05/xifontgan'
             '/checkpoints/Capitals64_cGAN/_log_train.txt')]
datas = []

loss_ = []
losses = []
iters = 0
for fp in fps:
    loss_ = []
    for line in fp:
        # line = '(epoch: 600, iters: 9120, time: 0.120) G_GAN: 0.002 G_L1: 13.827 G_MSE: 18.423 D_real: 0.002 D_fake: 0.401 Gsk1_L1: 39.687 Gsk1_MSE: 18.640 Gsk1_GAN: 0.370 Dsk1_real: 0.761 Dsk1_fake: 0.429'
        end_of_epoch = re.findall(r'End of epoch (\d+) / 600', line)
        # print(end_of_epoch)
        if end_of_epoch:
            # print(end_of_epoch)
            loss_ = loss_ + [[abs(x / iters) for x in losses]]
            losses = []
            iters = 0
        else:
            line_ = re.findall(r'\(epoch:.*', line)
            if not line_:
                continue
            losses_ = re.findall(r' (\d+\.\d*|\d+)', line_[0])
            # print(losses_)
            # re.split('[:|(|)|,]? ', line_[0])
            nbatch = int(losses_[1]) - iters
            iters = int(losses_[1])
            if not losses:
                losses = [float(x) * nbatch for x in losses_[3:]]
            else:
                losses = [x + float(y) * nbatch
                          for x, y in zip(losses, losses_[3:])]

    # print('[' + ', '.join(['%7.4f' % x for x in loss[599]]) + ']')

    loss = list(map(list, zip(*loss_)))

    data = {'G_GAN': loss[0],
            'G_L1': loss[1],
            'G_MSE': loss[2],
            'D_real': loss[3],
            # 'D_fake': loss[4],
            # 'Gsk1_L1': loss[5],
            # 'Gsk1_MSE': loss[6],
            # 'Gsk1_GAN': loss[7],
            # 'Dsk1_real': loss[8],
            # 'Dsk1_fake': loss[9],
            'epochs': [x + 1 for x in range(int(end_of_epoch[0]))]}
    datas += data
    plt.plot('G_L1', data=data)
    print(data['G_L1'])
    print(fp)


plt.ylabel('some numbers')
plt.savefig('temp.pdf')

# epoch = re.findall(r'End of epoch (\d+) / 1000', 'End of epoch 10 / 1000 	 Time Taken: 29 sec')

# epoch = re.findall(r'End of epoch (\d+) / (\d+)', 'End of epoch 10 / 1000 	 Time Taken: 29 sec')

# epoch = re.findall(r'(\d+|\d\.\d*)', '(epoch: 1, iters: 2400, time: 0.019) G_GAN: 1.154 G_L1: 31.157 D_real: 0.239 D_fake: 0.364 ')

# losses_temp = re.findall(r' (\d+\.\d*|\d+)', '(epoch: 600, iters: 9120, time: 0.120) G_GAN: 0.002 G_L1: 13.827 G_MSE: 18.423 D_real: 0.002 D_fake: 0.401 Gsk1_L1: 39.687 Gsk1_MSE: 18.640 Gsk1_GAN: 0.370 Dsk1_real: 0.761 Dsk1_fake: 0.429')

# [a + b for a, b in zip(list1, list2)]
# [re.split(":? ", entry, 4) for entry in entries]


def fp2data(fp, smooth=1):
    losses = []
    iters = 0
    loss_ = []
    for line in fp:
        # line = '(epoch: 600, iters: 9120, time: 0.120) G_GAN: 0.002 G_L1: 13.827 G_MSE: 18.423 D_real: 0.002 D_fake: 0.401 Gsk1_L1: 39.687 Gsk1_MSE: 18.640 Gsk1_GAN: 0.370 Dsk1_real: 0.761 Dsk1_fake: 0.429'
        end_of_epoch = re.findall(r'End of epoch (\d+) / (\d+)', line)
        if end_of_epoch:
            print('end of epoch %s' % end_of_epoch)
        if end_of_epoch:
            # print(end_of_epoch)
            loss_ = loss_ + [[end_of_epoch[0]] + [abs(x / iters) for
                                                 x in losses]]
            iters = 0
            losses = []
        else:
            line_ = re.findall(r'\(epoch:.*', line)
            if not line_:
                continue
            losses_ = re.findall(r' (\d+\.\d*|\d+)', line_[0])
            # print(losses_)
            # re.split('[:|(|)|,]? ', line_[0])
            iters_ = int(losses_[1])
            if iters_ < iters:
                if smooth:
                    iters = 0
                    losses = []
            nbatch = iters_ - iters
            iters = iters_
            if not losses:
                losses = [float(x) * nbatch for x in losses_[3:]]
            else:
                losses = [x + float(y) * nbatch
                          for x, y in zip(losses, losses_[3:])]

    # print('[' + ', '.join(['%7.4f' % x for x in loss[599]]) + ']')

    loss = list(map(list, zip(*loss_)))

    data = {'epoch_tuple':loss[0],
            'G_GAN': loss[1],
            'G_L1': loss[2],
            'G_MSE': loss[3],
            'D_real': loss[4]
            # 'D_fake': loss[4],
            # 'Gsk1_L1': loss[5],
            # 'Gsk1_MSE': loss[6],
            # 'Gsk1_GAN': loss[7],
            # 'Dsk1_real': loss[8],
            # 'Dsk1_fake': loss[9]
            }
    return data


fp_cGAN500_100 = open('/mnt/Files/XIremote/OneDrive - Wayne State University/XIcodes/Python/python3.6/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05/xifontgan/checkpoints/Capitals64_cGAN/_log_train.txt')
datas_cGAN500_100 = fp2data(fp_cGAN500_100)
data_cGAN500_100 = {key:value[0:500]+value[1491:] for key, value in
                    datas_cGAN500_100.items()}

fp_EcGAN500_100 = open('/mnt/Files/XIremote/OneDrive - Wayne State '
                       'University/XIcodes/Python/python3.6'
                       '/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05'
                       '/xifontgan/checkpoints/Capitals64_EcGAN'
                       '/_log_train.txt')
datas_EcGAN500_100 = fp2data(fp_EcGAN500_100)
data_EcGAN500_100 = {key:value[66:] for key, value in
                    datas_EcGAN500_100.items()}


fp_sk1GAN500_100 = open('/mnt/Files/XIremote/OneDrive - Wayne State '
                       'University/XIcodes/Python/python3.6'
                        '/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05/xifontgan/checkpoints/Capitals64_sk1GAN/_log_train.txt')
datas_sk1GAN500_100 = fp2data(fp_sk1GAN500_100)
data_sk1GAN500_100 = {key:value[98:] for key, value in
                      datas_sk1GAN500_100.items()}


fp_EskGAN500_100 = open('/mnt/Files/XIremote/OneDrive - Wayne State '
                       'University/XIcodes/Python/python3.6'
                        '/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05/xifontgan/checkpoints/Capitals64_EskGAN2_dspostG=1/_log_train.txt')
datas_EskGAN500_100 = fp2data(fp_EskGAN500_100)
data_sk1GAN500_100 = {key:value[98:] for key, value in
                      datas_sk1GAN500_100.items()}


plt.close()
plt.plot('G_L1', data=data_cGAN500_100, label='cGAN')
plt.plot('G_L1', data=data_sk1GAN500_100, label='skGAN')
plt.plot('G_L1', data=datas_EskGAN500_100, label='EskGAN')
plt.plot('G_L1', data=data_EcGAN500_100, label='EcGAN')
plt.ylabel('L1 losses')
plt.legend()
plt.savefig('L1 losses.pdf')
