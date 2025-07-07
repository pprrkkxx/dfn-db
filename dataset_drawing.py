import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import gc
from scipy.signal import welch
def find_data(file_path, modulation_type, snr, group_index):


        indices = [(i, j) for i, (mod_type, snr_value) in enumerate(zip(Y_data, Z_data))
                for j in range(len(mod_type)) if mod_type[j] == modulation_type and snr_value[j] == snr]

        if not indices:
            print("未找到指定调制方式和信噪比的数据。")
            return None

        selected_index = indices[group_index] if group_index < len(indices) else indices[-1]
        selected_data = X_data[selected_index[0]]
        return selected_data



def generate_noisy_analog_signal(dataset, mod_type, snr, n, carrier_offset, Fs):
    data = find_data(dataset, mod_type, snr, n)

    if data is not None:
        
        real_signal = data[0, :]
        imaginary_signal = data[1, :]
        analog_signal = real_signal + 1j * imaginary_signal
        # 可视化：原始时域信号
        plt.figure(figsize=(6, 4))
        plt.plot(np.real(analog_signal))  # 只显示前1000个点避免过密
        # 设置标题和坐标轴标签的字体大小
        plt.title("Original Time-domain Signal (Real Part)", fontsize=16)
        plt.xlabel("Sample Index", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)

        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 添加网格并适当缩小线宽
        plt.grid(True, linewidth=0.5)

        # 紧凑布局，让文字别被切掉
        plt.tight_layout()
        # plt.axis('off')

        plt.savefig('./fig_time_domain_original.png',bbox_inches='tight', pad_inches=0.1,dpi = 400)
        plt.close()


        del data
        del real_signal
        del imaginary_signal
        SNR_dB = -6
        SNR = 10 ** (SNR_dB / 10)

        N_upsample = 100
        N_pad = 10
        Len_Data = len(analog_signal)
        Len_signal = Len_Data * N_upsample
        Len_signal_pad = Len_signal * N_pad

        fs = 200000
        Fs = fs * N_upsample

        Data1 = np.zeros(Len_signal, dtype=analog_signal.dtype)
        Data1[::N_upsample] = analog_signal
        N = len(Data1)
        freq_axis = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))

        # 可视化：上采样后的信号
        plt.figure(figsize=(6, 4))
        plt.plot(np.real(Data1))  # Data1 是上采样后的信号
        plt.title("Upsampled Signal (Real Part)", fontsize=16)
        plt.xlabel("Sample Index", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
                # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linewidth=0.5)
        plt.tight_layout()
        plt.savefig("./fig_upsampled_signal.png", bbox_inches='tight', pad_inches=0.1,dpi = 400)
        plt.close()

        del analog_signal
        fft_Data1 = np.fft.fft(Data1)
        fft_before_noise = np.fft.fftshift(fft_Data1)  # 替换成实际变量
        plt.figure(figsize=(6, 4))
        plt.plot(freq_axis, np.abs(fft_before_noise))
        plt.title("FFT Spectrum Before Filtering", fontsize=16)
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel("Magnitude", fontsize=14)
                # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linewidth=0.5)
        plt.tight_layout()
        plt.savefig("./fig_fft_before_noise.png", bbox_inches='tight', pad_inches=0,dpi = 400)
        plt.close()

        del Data1
        Pdf_sn = np.mean(fft_Data1 * np.conj(fft_Data1))
        Pdf_n = Pdf_sn / (1 + SNR)
        # 频域加噪声
        Pos_noise = np.round(np.arange(Len_Data / 2, Len_signal - Len_Data / 2))
        fft_Data1[Pos_noise.astype(int)] = 0
        # 可视化：FFT频谱（对数尺度）
        fft_after_filter = np.fft.fftshift(fft_Data1)  # 替换成实际变量
        plt.figure(figsize=(6, 4))
        plt.plot(freq_axis, np.abs(fft_after_filter))
        plt.title("FFT Spectrum After Filtering", fontsize=16)
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel("Magnitude", fontsize=14)
                # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linewidth=0.5)
        plt.tight_layout()
        plt.savefig("./fig_fft_after_filter.png", bbox_inches='tight', pad_inches=0,dpi = 400)
        plt.close()

        fft_Data1[Pos_noise.astype(int)] = np.sqrt(Pdf_n / 2) * (np.random.randn(len(Pos_noise)) + 1j * np.random.randn(len(Pos_noise)))
        # # 生成复高斯噪声
        # noise = np.sqrt(Pdf_n / 2) * (np.random.randn(len(Pos_noise)) + 1j * np.random.randn(len(Pos_noise)))

        # # 添加到正频率
        # fft_Data1[Pos_noise.astype(int)] = noise

        # # 映射负频率索引：负频率 = N - 正频率
        # N = len(fft_Data1)
        # Neg_noise = N - Pos_noise.astype(int)

        # # 添加共轭到负频率
        # fft_Data1[Neg_noise] = np.conj(noise)

        # plt.figure()
        # f, Pxx = welch(np.fft.ifft(100*fft_Data1), fs=Fs, nperseg=1024, return_onesided=False)
        # plt.semilogy(f, Pxx)
        # plt.title("Power Spectral Density After Noise Injection")
        # # plt.axis('off')
        # plt.savefig("./fig_psd_after_noise.png", bbox_inches='tight', pad_inches=0)
        # plt.close()


        # === 3. 加噪后频谱图 ===
        fft_after_noise = np.fft.fftshift(fft_Data1)  # 替换成实际变量
        plt.figure(figsize=(6, 4))
        plt.plot(freq_axis, np.abs(fft_after_noise))
        plt.title("FFT Spectrum After Noise Injection", fontsize=16)
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel("Magnitude", fontsize=14)
                # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linewidth=0.5)
        plt.tight_layout()
        plt.savefig("./fig_fft_after_noise.png", bbox_inches='tight', pad_inches=0,dpi = 400)
        plt.close()

        
        signal = np.fft.ifft(fft_Data1)
        # === 2. 加噪后时域图 ===
        plt.figure(figsize=(6, 4))
        plt.plot(np.real(signal))
        plt.title("Time-domain Signal After Noise Injection", fontsize=16)
        plt.xlabel("Sample Index", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
                # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linewidth=0.5)
        plt.tight_layout()
        plt.savefig("./fig_time_domain_noisy.png", bbox_inches='tight', pad_inches=0,dpi = 400)
        plt.close()

        del fft_Data1
        t = np.arange(Len_signal) / Fs
        signal = signal * np.exp(1j * 2 * np.pi * carrier_offset * t)
        # 可视化：频移后的频域信号

        # plt.figure()
        # f_shift, Pxx_shift = welch(signal, fs=Fs, nperseg=1024, return_onesided=False)
        # plt.semilogy(f_shift, Pxx_shift)
        # plt.title("Power Spectral Density After Frequency Shift")
        # # plt.axis('off')
        # plt.savefig("./fig_psd_after_freq_shift.png", bbox_inches='tight', pad_inches=0)
        # plt.close()

        # === 1. 频移后频谱图 ===
        fft_shifted = np.fft.fftshift(np.fft.fft(signal))
        plt.figure(figsize=(6, 4))
        plt.plot(freq_axis, np.abs(fft_shifted))
        plt.title("Spectrum After Carrier Frequency Offset", fontsize=16)
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel("Magnitude", fontsize=14)
                # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linewidth=0.5)
        plt.tight_layout()
        plt.savefig("./fig_fft_after_freq_shift.png", bbox_inches='tight', pad_inches=0,dpi = 400)
        plt.close()




        return signal

def generate_and_save_spectrogram(file_path, modulation_type,modulation_type1,modulation_type2,modulation_type3,modulation_type4,modulation_type5, snr1, save_filename1,save_filename2):
    # 调用函数生成带噪声的模拟信号
    n1 = np.random.randint(0, 900)
    n2 = np.random.randint(0, 900)
    n3 = np.random.randint(0, 900)
    n4 = np.random.randint(0, 900)
    n5 = np.random.randint(0, 900)
    n6 = np.random.randint(0, 900)
    f1 = np.random.randint(random_f, -random_f)
    f2 = np.random.randint(random_f, -random_f)
    f3 = np.random.randint(random_f, -random_f)
    f4 = np.random.randint(random_f, -random_f)
    f5 = np.random.randint(random_f, -random_f)
    f6 = np.random.randint(random_f, -random_f)

    analog_signal = generate_noisy_analog_signal(file_path, modulation_type, snr1, n1, f1, Fs)
    analog_signal1 = generate_noisy_analog_signal(file_path, modulation_type1, snr1, n2,f2, Fs)
    analog_signal2 = generate_noisy_analog_signal(file_path, modulation_type2, snr1, n3, f3, Fs)
    analog_signal3 = generate_noisy_analog_signal(file_path, modulation_type3, snr1, n4, f4, Fs)
    analog_signal4 = generate_noisy_analog_signal(file_path, modulation_type4, snr1, n5, f5, Fs)
    analog_signal5 = generate_noisy_analog_signal(file_path, modulation_type5, snr1, n6, f6, Fs)

    orlenth=len(analog_signal)

    #裁剪信号
    sl1=np.random.randint(len(analog_signal) // 3, len(analog_signal))
    sl2=np.random.randint(len(analog_signal) // 3, len(analog_signal))
    sl3=np.random.randint(len(analog_signal) // 3, len(analog_signal))
    sl4=np.random.randint(len(analog_signal) // 3, len(analog_signal))
    sl5=np.random.randint(len(analog_signal) // 3, len(analog_signal))
    sl6=np.random.randint(len(analog_signal) // 3, len(analog_signal))

    analog_signal = analog_signal[:sl1]
    analog_signal1 = analog_signal1[:sl2]
    analog_signal2 = analog_signal2[:sl3]
    analog_signal3 = analog_signal3[:sl4]
    analog_signal4 = analog_signal4[:sl5]
    analog_signal5 = analog_signal5[:sl6]




    las1 = len(analog_signal)
    las2 = len(analog_signal1)
    las3 = len(analog_signal2)
    las4 = len(analog_signal3)
    las5 = len(analog_signal4)
    las6 = len(analog_signal5)
    las_all = las1+las2+las3+las4+las5+las6

    nfft = 256
    window = np.hamming(nfft)
    overlap = nfft // 2

    # 计算信号的功率
    signal_power = np.mean(np.abs(analog_signal)**2)
    # print(SNR)

    # 计算噪声的功率
    noise_power = signal_power / 10 ** (SNR / 10.0)


    # 生成高斯白噪声 
    
    random_length1 = np.random.randint(orlenth // 4, orlenth//2)
    random_length2 = np.random.randint(orlenth// 4, orlenth//2)
    random_length3 = np.random.randint(orlenth // 4, orlenth//2)
    random_length4 = np.random.randint(orlenth // 4, orlenth//2)
    random_length5 = np.random.randint(orlenth // 4, orlenth//2)
    random_length6 = np.random.randint(orlenth // 4, orlenth//2)
    ns = random_length1 + random_length2 + random_length3 + random_length4 + random_length5 +random_length6
    random_length7 = int(7.2*orlenth)- ns -las_all
    if random_length7 < 0:
         random_length7 = 1 
    # print(random_length6)
    # noise1 = np.random.normal(0, np.sqrt(noise_power), random_length1)
    # noise2 = np.random.normal(0, np.sqrt(noise_power), random_length2)
    # noise3 = np.random.normal(0, np.sqrt(noise_power), random_length3)
    # noise4 = np.random.normal(0, np.sqrt(noise_power), random_length4)
    # noise5 = np.random.normal(0, np.sqrt(noise_power), random_length5)
    # noise6 = np.random.normal(0, np.sqrt(noise_power), random_length6)
    # noise7 = np.random.normal(0, np.sqrt(noise_power), random_length7)
    # 生成复数高斯白噪声（实部和虚部独立）
    noise1 = np.random.normal(0, np.sqrt(noise_power/2), random_length1) + 1j * np.random.normal(0, np.sqrt(noise_power/2), random_length1)
    noise2 = np.random.normal(0, np.sqrt(noise_power/2), random_length2) + 1j * np.random.normal(0, np.sqrt(noise_power/2), random_length2)
    noise3 = np.random.normal(0, np.sqrt(noise_power/2), random_length3) + 1j * np.random.normal(0, np.sqrt(noise_power/2), random_length3)
    noise4 = np.random.normal(0, np.sqrt(noise_power/2), random_length4) + 1j * np.random.normal(0, np.sqrt(noise_power/2), random_length4)
    noise5 = np.random.normal(0, np.sqrt(noise_power/2), random_length5) + 1j * np.random.normal(0, np.sqrt(noise_power/2), random_length5)
    noise6 = np.random.normal(0, np.sqrt(noise_power/2), random_length6) + 1j * np.random.normal(0, np.sqrt(noise_power/2), random_length6)
    noise7 = np.random.normal(0, np.sqrt(noise_power/2), random_length7) + 1j * np.random.normal(0, np.sqrt(noise_power/2), random_length7)


    
    #模拟重合情况
    print('最大长度为：',7.2*orlenth)
    
    

    noisy_analog_signal = np.concatenate((noise1, analog_signal, noise2, analog_signal1, noise3 ,analog_signal2,noise4 ,analog_signal3,noise5,analog_signal4,noise6,analog_signal5,noise7))
    N1 = len(noisy_analog_signal)
    freq_axis1 = np.fft.fftshift(np.fft.fftfreq(N1, d=1/Fs))
                # === 1. 频移后频谱图 ===
    fft_shifted = np.fft.fftshift(np.fft.fft(noisy_analog_signal))
    plt.figure(figsize=(6, 4))
    plt.plot(freq_axis1, np.abs(fft_shifted))
    plt.title("Signals Spectrum", fontsize=16)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
                # 设置刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig("./fig_fft_all.png", bbox_inches='tight', pad_inches=0,dpi = 400)
    plt.close()
    print('现在长度为：',len(noisy_analog_signal))
    if len(noisy_analog_signal)>737280:
         print('大于737280,结束本次循环')
         return None
         

    nas = len(noisy_analog_signal)



    
    ################################################################################################################################
    # 计算信号的起始位置
    start_time_1 = len(noise1)/ nas
    end_time_1 = start_time_1 + las1 / nas
    c1 = (start_time_1 + end_time_1)/2
    l1 = (end_time_1 - start_time_1)
    

    start_time_2 = end_time_1+ len(noise2) / nas
    end_time_2 = start_time_2 + las2 / nas
    c2 = (start_time_2 + end_time_2)/2
    l2 = (end_time_2 - start_time_2)
    

    start_time_3 = end_time_2 + len(noise3) / nas
    end_time_3 = start_time_3 + las3 / nas
    c3 = (start_time_3 + end_time_3)/2
    l3 = (end_time_3 - start_time_3)
    

    start_time_4 = end_time_3 + len(noise4) / nas
    end_time_4 = start_time_4 + las4 / nas
    c4 = (start_time_4 + end_time_4)/2
    l4 = (end_time_4 - start_time_4)

    start_time_5 = end_time_4 + len(noise5) / nas
    end_time_5 = start_time_5 + las5 / nas
    c5 = (start_time_5 + end_time_5)/2
    l5 = (end_time_5 - start_time_5)

    start_time_6 = end_time_5 + len(noise6) / nas
    end_time_6 = start_time_6 + las6 / nas
    c6 = (start_time_6 + end_time_6)/2
    l6 = (end_time_6 - start_time_6)
    # print(analog_signal)

    del analog_signal
    del analog_signal1
    del analog_signal2
    del analog_signal3
    del noise1
    del noise2
    del noise3
    del noise4
    del noise5

    c1 = "{:.6f}".format(c1)
    l1 = "{:.6f}".format(l1)
    c2 = "{:.6f}".format(c2)
    l2 = "{:.6f}".format(l2)
    c3 = "{:.6f}".format(c3)
    l3 = "{:.6f}".format(l3)
    c4 = "{:.6f}".format(c4)
    l4 = "{:.6f}".format(l4)
    c5 = "{:.6f}".format(c5)
    l5 = "{:.6f}".format(l5)
    c6 = "{:.6f}".format(c6)
    l6 = "{:.6f}".format(l6)


    F_all = 10039062.5 + 9960937.5
    # 计算信号的功率谱密度


    # 计算第一个信号的中心频率

    center_frequency_1 = (9960937.5 + f1 + 10000)/F_all
    center_frequency_2 = (9960937.5 + f2 + 10000)/F_all
    center_frequency_3 = (9960937.5 + f3 + 10000)/F_all
    center_frequency_4 = (9960937.5 + f4 + 10000)/F_all
    center_frequency_5 = (9960937.5 + f5 + 10000)/F_all
    center_frequency_6 = (9960937.5 + f6 + 10000)/F_all

    # 计算带宽

    bw_1 = 0.034000
    bw_2 = 0.034000
    bw_3 = 0.034000
    bw_4 = 0.034000
    bw_5 = 0.034000
    bw_6 = 0.034000

    center_frequency_1 = "{:.6f}".format(center_frequency_1)
    center_frequency_2 = "{:.6f}".format(center_frequency_2)
    center_frequency_3 = "{:.6f}".format(center_frequency_3)
    center_frequency_4 = "{:.6f}".format(center_frequency_4)
    center_frequency_5 = "{:.6f}".format(center_frequency_5)
    center_frequency_6 = "{:.6f}".format(center_frequency_6)
    bw_1 = "{:.6f}".format(bw_1)
    bw_2 = "{:.6f}".format(bw_2)
    bw_3 = "{:.6f}".format(bw_3)
    bw_4 = "{:.6f}".format(bw_4)
    bw_5 = "{:.6f}".format(bw_5)
    bw_6 = "{:.6f}".format(bw_6)

    modulation_type0=0
    # with open('./txt/{}'.format(save_filename2), 'w') as file:
    #     file.write(f'{modulation_type} {c1} {center_frequency_1} {l1} {bw_1}\n')
    #     file.write(f'{modulation_type1} {c2} {center_frequency_2} {l2} {bw_2}\n')
    #     file.write(f'{modulation_type2} {c3} {center_frequency_3} {l3} {bw_3}\n')
    #     file.write(f'{modulation_type3} {c4} {center_frequency_4} {l4} {bw_4}\n')
    # file.close()
    # with open('./能量检测数据集/0db/{}'.format(save_filename2), 'w') as file:
    #     file.write(f'{modulation_type0} {c1} {center_frequency_1} {l1} {bw_1}\n')
    #     file.write(f'{modulation_type0} {c2} {center_frequency_2} {l2} {bw_2}\n')
    #     file.write(f'{modulation_type0} {c3} {center_frequency_3} {l3} {bw_3}\n')
    #     file.write(f'{modulation_type0} {c4} {center_frequency_4} {l4} {bw_4}\n')
    #     file.write(f'{modulation_type0} {c5} {center_frequency_5} {l5} {bw_5}\n')
    #     file.write(f'{modulation_type0} {c6} {center_frequency_6} {l6} {bw_6}\n')
    # file.close()

    #################################################################################################################################
    matplotlib.use("Agg")
    _, frequencies, times, spectrogram = plt.specgram(noisy_analog_signal, NFFT=nfft, Fs=Fs, window=window, noverlap=overlap)
    #
    #
    #
    #
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    # 绘制时频图
    pcm = ax.pcolormesh(times, frequencies, spectrogram.get_array())
    # plt.colorbar(pcm, ax=ax)
    ax.axis('off')

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #
    #
    # 保存图像
    plt.savefig('./{}'.format(save_filename1), bbox_inches='tight', pad_inches=0)
    fig.clf()
    ax.cla()
    plt.close(fig)
    del times
    del frequencies
    del spectrogram
    return noisy_analog_signal


# generate_and_save_spectrogram(file_path, modulation_type,modulation_type1,modulation_type2,modulation_type3, snr1,snr2, save_filename1,save_filename2)
num_iterations = 500
file_path = 'train_mat.h5'
Fs = 20000000 # 升采样率
# number=[0,1,7,8,13,14,15,16,17,18,19,24]
# number=[0,7,13,14,15,16,17,18,19,20,21,22,23,24]
number=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]


# # 指定要保存的H5文件路径和名称

h5_file_path = './能量检测数据集/0db/0dB.h5'
with h5py.File(file_path, 'r') as file:
    X_data = file['X'][:]
    Y_data = file['Y'][:]
    Z_data = file['Z'][:]
    # print(Z_data.shape)
    # print(Y_data.shape)

# # 创建H5文件
# with h5py.File(h5_file_path, 'w') as hf:
#     # 创建 'X'、'Y'、'Z' 三个空键值
#     x_dataset=hf.create_dataset('X', shape=(0, 737280), maxshape=(None, 737280), dtype=np.complex128)
    # x_dataset = hf.create_dataset('X', shape=(num_iterations, 819200), dtype=np.complex128)
#     y_dataset = hf.create_dataset('Y', shape=(num_iterations, 4, 1), dtype=float)
#     z_dataset = hf.create_dataset('Z', shape=(num_iterations, 1), dtype=float)
skip = 0
for i in range(1):
    # modulation_type = i//205
    # modulation_type1 = i//205
    # modulation_type2 = i//205
    # modulation_type3 = i//205
    # n = i //400
    # modulation_type = number[n]
    # modulation_type1 = number[n]
    # modulation_type2 = number[n]
    # modulation_type3 = number[n]
    modulation_type = np.random.choice(number)
    modulation_type1 = np.random.choice(number)
    modulation_type2 = np.random.choice(number)
    modulation_type3 = np.random.choice(number)
    modulation_type4 = np.random.choice(number)
    modulation_type5 = np.random.choice(number)
    print('现在是几个循环：',i)

    # random_number = np.random.randint(-5, 9)
    # random_number1 = np.random.randint(-5, 9)
    snr1 = -6  # random_number * 2
        # random_number1 * 2
    SNR = 0

    name=i-skip

    save_filename1 = f'{name:012d}.png'  # 根据循环次数更改文件名
    save_filename2 = f'{name:012d}.txt'  # 根据循环次数更改文件名
    random_f = -8000000

    noisy_analog_signal = generate_and_save_spectrogram(file_path, modulation_type, modulation_type1,
                                                        modulation_type2, modulation_type3,modulation_type4,modulation_type5, snr1,
                                                        save_filename1, save_filename2)
    if noisy_analog_signal is None:  # 假设函数在条件满足时返回None
        print('主循环跳过本轮')
        skip = skip+1
        continue
    # print(noisy_analog_signal.shape)



    # x_dataset[i, :] = noisy_analog_signal
    # print(x_dataset[i, :])

    # current_size = hf['X'].shape[0]
    # hf['X'].resize(current_size + 1, axis=0)
    # hf['X'][current_size, :] = noisy_analog_signal
    # print('录入多少组数据',name)

    del noisy_analog_signal
    #
    # y_dataset[i, 0, :] = modulation_type
    # y_dataset[i, 1, :] = modulation_type1
    # y_dataset[i, 2, :] = modulation_type2
    # y_dataset[i, 3, :] = modulation_type3
    #
    # z_dataset[i, :] = snr1

    # process = psutil.Process(os.getpid())
    # print(f"memory:{process.memory_info().rss / 1024 ** 2} MB")
    # gc.collect()
    # with h5py.File('mo_ty.h5', 'w') as f:
    #     y_data = f.create_dataset('Y', shape=(num_iterations*4, ), dtype=float)
    #     for j in range(num_iterations):
    #         y_data[4*j]=y_dataset[j, 0, :]
    #         y_data[4*j+1 ]= y_dataset[j, 1, :]
    #         y_data[4*j + 2] = y_dataset[j, 2, :]
    #         y_data[4*j + 3] = y_dataset[j, 3, :]
file.close()


