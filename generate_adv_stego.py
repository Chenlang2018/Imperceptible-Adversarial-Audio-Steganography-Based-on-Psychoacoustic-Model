import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
import librosa
import generate_masking_threshold as generate_mask
import model
import random
import time


window_size = 2048
length = 16384
initial_bound = 1000
batch_size = 1
lr_stage1 = 0.1
lr_stage2 = 0.0001
num_iter_stage1 = 100
num_iter_stage2 = 300
positive = torch.ones((batch_size, 1))
negative = torch.zeros((batch_size, 1))
s_criterion = nn.BCELoss(reduction='none')
gradients = torch.ones((batch_size, length))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_ids = [0, 1, 2, 3]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ReadFromWav(data_dir, batch_size):
    """
    Returns:
        audios_np: a numpy array of size (batch_size, max_length) in float
        cover_labels: a numpy array of cover labels (batch_size, )
        th_batch: a numpy array of the masking threshold, each of size (?, 1025)
        psd_max_batch: a numpy array of the psd_max of the original audio (batch_size)
        length: the length of each audio sample in dataset batch
        sample_rate: int number
    """
    global sample_rate
    audios = []
    th_batch = []
    psd_max_batch = []
    # read the wav file
    for i in range(batch_size):
        sample_rate, wave_data = wave.read(str(data_dir[i]))
        audios.append(wave_data)
    audios = np.array(audios).reshape((batch_size, length)).astype(np.float32)
    # compute the masking threshold
    for i in range(batch_size):
        th, psd_max = generate_mask.generate_th(audios[i], sample_rate, window_size)
        th_batch.append(th)
        psd_max_batch.append(psd_max)
    th_batch = np.array(th_batch)
    psd_max_batch = np.array(psd_max_batch)
    # set the labels for cover audio
    cover_labels = torch.ones((batch_size, 1)).to(device)
    return audios, cover_labels, th_batch, psd_max_batch, sample_rate


# LSBM steganography
def embedding(cover_audio):
    stego_audio = []
    for i in range(batch_size):
        cover = cover_audio[i].reshape(16384)
        cover = cover.astype(np.int16)
        L = 16384
        stego = cover
        msg = np.random.randint(0, 2, L)
        msg = np.array(msg)
        k = np.random.randint(0, 2, L)
        k = np.array(k)
        for j in range(L):
            x = abs(cover[j])
            x = bin(x)
            x = x[2:]
            y = msg[j]
            if str(y) == x[-1]:
                stego[j] = cover[j]
            else:
                if k[j] == 0:
                    stego[j] = cover[j] - 1
                else:
                    stego[j] = cover[j] + 1
        stego = stego.reshape(16384)
        stego_audio.append(stego)
    stego_audio = np.array(stego_audio).reshape((batch_size, length))
    return torch.tensor(stego_audio).type(torch.FloatTensor)


def transform(x, window_size, psd_max_ori):
    scale = 8. / 3.
    n_fft = window_size
    hop_length = int(window_size // 4)
    win_length = window_size
    window_size = window_size

    win = librosa.core.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,center=False)
    z = scale * np.abs(win.T / window_size)
    psd = np.square(z)
    PSD = pow(10., 9.6) / psd_max_ori * psd
    return PSD


def compute_loss_th(delta, window_size, th_batch, psd_max_batch):
    loss_th_list =[]
    for i in range(batch_size):
        logits_delta = transform(delta[i, :], window_size, psd_max_batch[i])
        f = torch.nn.ReLU()
        loss_th = f(torch.from_numpy(logits_delta - th_batch[i])).mean()
        loss_th_list.append(loss_th)
    loss_th = torch.tensor(loss_th_list).reshape((batch_size, 1)).type(torch.FloatTensor)
    return loss_th


def attack_stage1(audios, steganalyzer, cover_labels, length, lr_stage1):
    delta = torch.zeros((batch_size, length), requires_grad=True)
    final_adv1 = torch.zeros((batch_size, length))
    optimizer1 = torch.optim.Adam([delta], lr=lr_stage1)
    for i in range(num_iter_stage1):
        new_input = delta + audios
        new_input_clip = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)
        new_input_stego = new_input_clip.reshape((-1, 1, length))
        stego_output = steganalyzer(new_input_stego)
        predict_labels = torch.where(stego_output.cpu().data > 0.5, positive, negative).to(device)
        bce_loss = s_criterion(stego_output, cover_labels).to(device)

        optimizer1.zero_grad()
        bce_loss.backward()
        delta.grad = torch.sign(delta.grad)
        optimizer1.step()

        bce_loss_output = bce_loss.item()
        delta_out_put = delta.data

        for ii in range(batch_size):
            delta.data[ii] = torch.clamp(delta.data[ii], -initial_bound, initial_bound)
        for ii in range(batch_size):
            if i % 5 == 0:
                if predict_labels[ii] == cover_labels[ii]:
                    print('=======================================True=======================================\n')
            final_adv1[ii]=new_input_clip[ii]
            print('Iteration [{}/{}], bce_loss: {}, '
                  'delta: {}'.format(ii+1, i+1, bce_loss_output, delta_out_put))
            if (i == num_iter_stage1 -1 and (final_adv1[ii] == 0).all()):
                final_adv1[ii] = new_input_clip[ii]
    return final_adv1


def attack_stage2(audios, steganalyzer, cover_labels, adv_distortion, th_batch, psd_max_batch, lr_stage2):
    delta = adv_distortion.clone().detach().requires_grad_(True)
    th_loss = torch.tensor([[np.inf] * batch_size]).reshape((batch_size, 1)).to(device)
    alpha = torch.ones((batch_size, 1)) * 0.05
    alpha = alpha.to(device)
    final_alpha = torch.zeros(batch_size)
    final_adv2 = torch.zeros((batch_size, length))
    optimizer2 = torch.optim.Adam([delta], lr=lr_stage2)
    min_th = -np.inf
    for i in range(num_iter_stage2):
        new_input = delta + audios
        new_input_clip = torch.clamp(new_input, -2 ** 15, 2 ** 15 - 1)
        new_input_stego = new_input_clip.reshape((-1, 1, length))
        stego_output = steganalyzer(new_input_stego)
        predict_labels = torch.where(stego_output.cpu().data > 0.5, positive, negative).to(device)
        bce_loss = s_criterion(stego_output, cover_labels).to(device)
        th_loss_temp = compute_loss_th(delta.cpu().detach().numpy(), window_size, th_batch, psd_max_batch).to(device)
        total_loss = bce_loss + alpha * th_loss_temp

        optimizer2.zero_grad()
        total_loss.backward()
        optimizer2.step()

        th_loss_output = th_loss_temp.cpu().detach().numpy()
        alpha_output = alpha.cpu().detach().numpy()

        for ii in range(batch_size):
            if predict_labels[ii] == cover_labels[ii]:
                if th_loss_temp[ii] < th_loss[ii]:
                    th_loss[ii] = th_loss_temp[ii]
                    final_alpha[ii] = alpha[ii]
                    final_adv2[ii] = new_input_clip[ii]
                    print('==============================Attack Succeed!==============================')
            if i % 20 ==0:
                alpha[ii] *= 1.2
            if i % 20 == 0 and predict_labels[ii] != cover_labels[ii]:
                alpha[ii] *= 0.8
                alpha[ii] = max(alpha[ii], min_th)
            print('Iteration [{}/{}], th_loss: {}, '
                  'alpha: {}'.format(ii+1, i+1, th_loss_output[ii], alpha_output[ii]))
            if (i == num_iter_stage2 -1 and (final_adv2[ii] == 0).all()):
                final_adv2[ii] = new_input_clip[ii]
    return final_adv2, th_loss, final_alpha


def main():
    steganalyzer = torch.nn.DataParallel(model.Steganalyzer().to(device), device_ids=gpu_ids)
    steganalyzer = steganalyzer.cuda(device)
    steganalyzer.load_state_dict(torch.load('steganalyzer_trained.pth'))
    steganalyzer.eval()

    data_dir = np.loadtxt('data_dir3.txt', dtype=str, delimiter=",")
    audios, cover_labels, th_batch, psd_max_batch, sample_rate = ReadFromWav(data_dir,batch_size)
    stego = embedding(audios)
    # Attack for stage 1
    print('=============================================Attack for stage 1 started!=============================================\n')
    adv_example_stego1 = attack_stage1(stego, steganalyzer, cover_labels, length, lr_stage1)
    for i in range(batch_size):
        distortion1 = (adv_example_stego1[i] - stego[i]).cpu().detach().numpy()
        distortion1_max= np.max(distortion1)
        print('Sample [{}/{}], final distortion for stage 1: {:.6f}'.format(i+1, batch_size, distortion1_max))
        adv_example_stego1[i] = adv_example_stego1[i].reshape(length)
        temp = adv_example_stego1[i].cpu().detach().numpy().astype(np.int16)
        wave.write('./adv_stego1/{}_stage1.wav'.format(i + 1), 16000,  temp)

    # plot waveform and perturbation for stage 1

    t = np.arange(0, length) * (1.0 / length)
    plt.plot(t, audios[0])
    plt.plot(t, distortion1,'orange')
    plt.show()


    #Attack for stage 2
    print('=============================================Attack for stage 2 started!=============================================\n')
    adv = adv_example_stego1 - stego
    adv_example_stego2, th_loss, final_alpha = attack_stage2(stego, steganalyzer, cover_labels, adv, th_batch, psd_max_batch, lr_stage2)
    for i in range(batch_size):
        distortion2 = (adv_example_stego2[i] - stego[i]).cpu().detach().numpy()
        distortion2_max = np.max(distortion2)
        print('Sample [{}/{}], final distortion for stage 2: {:.6f}'.format(i+1, batch_size, distortion2_max))
        adv_example_stego2[i] = adv_example_stego2[i].reshape(length)
        temp= adv_example_stego2[i].cpu().detach().numpy().astype(np.int16)
        wave.write('./adv_stego2/{}_stage2.wav'.format(i + 1), 16000, temp)
    # plot waveform and perturbation for stage 2
    '''
    t = np.arange(0, length) * (1.0 / length)
    plt.plot(t, audios[0])
    plt.plot(t, distortion2, 'orange')
    plt.show()
    '''


if __name__ == '__main__':
    set_seed(1)
    start = time.time()
    main()
    end = time.time()
    print('Elapsed training time: {:.2f}min'.format((end - start) / 60))

