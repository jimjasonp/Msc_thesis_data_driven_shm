
'''
CONTENTS

1)X AND Y SET CREATORS

2)FOURIER SIGNAL NORMALIZATION, SIGNAL PROPERTIES AND HARMONICS EXTRACTION

3)DATA TRANSFORMATIONS

4)FEATURE ENGINEERING TECHNIQUES

5)PLOTS

6)EXPERIMENT RUN

7)TOOLS FOR TUNING

8)EXPERIMENTAL RESULTS EXTRACTION
'''



########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
1) X AND Y SET CREATORS

---> x set creator (X_set)
Takes as input the path and a transformation the number of points and the noise percentage. The outputs are five:the first is the concatenated amplitude of all three sensors
the second,third and fourth are the amplitudes of the second,third and fourth sensor, the fifth is the frequency 


---> y set creator (classification and regression) (y_set)
Takes as input the path. The output is a dataframe containing columns with elements of each sample. These elements are: damage percentage as 'dmg',
the filename as 'damage_file_name', the case study as 'caseStudey', the kind of defect as 'defect' and the index number as 'dmg_index_number'


'''
def X_set(path, transformation, n_points, noise_percent=None):
    '''
    transformations: 'none', 'fourier'
    - For 'none': returns truncated raw time-series (first n_points).
    - For 'fourier': returns truncated FFT amplitude and frequency vectors (first n_points).
    - Noise (if noise_percent is provided) is added before transformation.
    '''
    import os
    import glob
    import numpy as np
    import pandas as pd
    from helper_functions import fourier, add_noiz  # Ensure add_noiz is imported

    sensor_data_list = []
    name_list = []

    for filename in sorted(glob.glob(os.path.join(path, "data*"))):
        filename = filename.removesuffix('.csv')
        name_list.append(filename)

    sensor_data = pd.DataFrame({'name': name_list})
    sensor_data['sensor_index_number'] = [int(i.split('_')[-1]) for i in sensor_data['name']]
    sensor_data = sensor_data.sort_values(by='sensor_index_number')
    new_names = [name + '.csv' for name in sensor_data['name']]

    for filename in new_names:
        df = pd.read_csv(filename, sep=' |,', engine='python').dropna()
        sensor_data_list.append(df)

    freq_list = []
    power_spectrum_list = []
    sensor_names = ['s2', 's3', 's4']

    for sensor in sensor_names:
        for i in range(len(sensor_data_list)):
            sample_sensor = sensor_data_list[i][sensor].values[:n_points]

            if noise_percent is not None:
                sample_sensor = add_noiz(sample_sensor.reshape(1, -1), noise_percent).flatten()

            if transformation == 'fourier':
                amp, freq = fourier(sample_sensor)
                amp = amp[:n_points]
                power_spectrum = amp

                if sensor == 's2':  # Only append freq once per file
                    freq = freq[:n_points]
                    freq_list.append(freq)

            elif transformation == 'none':
                power_spectrum = sample_sensor

            power_spectrum_list.append(power_spectrum)

    num_samples = len(power_spectrum_list) // 3
    sensor2_vector = power_spectrum_list[0:num_samples]
    sensor3_vector = power_spectrum_list[num_samples:2 * num_samples]
    sensor4_vector = power_spectrum_list[2 * num_samples:3 * num_samples]

    X = np.concatenate((sensor2_vector, sensor3_vector, sensor4_vector), axis=1)

    return X, sensor2_vector, sensor3_vector, sensor4_vector, freq_list

def y_set(path):
    
    '''
    select column ['dmg'] which is the damage percentage for regression or ['defect'] which is the defect for classification
    
    '''
    import numpy as np
    import pandas as pd
    import os
    import glob

    dmg_list = []
    name_list = []
    case_list = []
    defect_list =[]
    for filename in glob.glob(os.path.join(path , "meta*")):
        df = pd.read_csv(filename,sep=' |,', engine='python')
        dmg_perc = df['Damage_percentage']
        case = df['caseStudey'][0]
        dmg_perc = dmg_perc[0]
        dmg_list.append(dmg_perc)
        filename = filename.removesuffix('.csv')
        
        df_defect = df['DamageLayer1'][0] + df['DamageLayer3'][0] + df['DamageLayer5'][0]
        dm_defect = df['DamageLayer1'][1] + df['DamageLayer3'][1] + df['DamageLayer5'][1]
        dd_defect = df['DamageLayer2'][0] + df['DamageLayer4'][0]
        
        if df_defect ==0 and dm_defect ==0 and dd_defect ==0:
            defect_list.append('clean')
        elif df_defect !=0 and dm_defect !=0 and dd_defect !=0:
            defect_list.append('all defect modes')
        elif df_defect !=0 and dm_defect ==0 and dd_defect ==0:
            defect_list.append('df')
        elif df_defect ==0 and dm_defect !=0 and dd_defect ==0:
            defect_list.append('dm')
        elif df_defect ==0 and dm_defect ==0 and dd_defect !=0:
            defect_list.append('dd')
        else:
            defect_list.append('all defect modes')
        
        name_list.append(filename)
        case_list.append(case)
 
    dmg_data = pd.DataFrame({'dmg':dmg_list,'damage_file_name':name_list,'caseStudey':case_list,'defect':defect_list})
    dmg_data['dmg_index_number'] = [int(i.split('_')[-1]) for i in dmg_data['damage_file_name']]
    dmg_data = dmg_data.sort_values(by=['dmg_index_number'])
    return dmg_data
    
########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


'''

2) FOURIER SIGNAL NORMALIZATION, SIGNAL PROPERTIES AND HARMONICS EXTRACTION

- - - A) FOURIER SIGNAL NORMALIZATION

- - - B) SIGNAL PROPERTIES EXTRACTION WITH NORMALIZATION

- - - C) HARMONICS WITH NORMALIZATION 

'''



############################################################

'''

A) FOURIER SIGNAL NORMALIZATION


the following functions work together
to get the normalized fft use fourier_nrm_vector
which takes as input the path and the output is the normalized fft

---> fourier signal normalization (fourier_signal_normalization)
The input is a sample of signal, it calculates its fft and it then is normalized according to the excitation frequency
The amplitude is normalized according to the amplitude of the excitation frequency and the frequency is normalized according to the excitation frequency

---> fourier vector maker (fourier_vector_maker)
The input is a list of signals and the function 'fourier_signal_normalization' is applied and calculates the normalized fft.
The output is a list of normalized amplitudes and a list of normalized frequencies.

---> fourier nrm vector (fourier_nrm_vector)
The input is the path of the file containing the data. The function 'fourier_vector_maker' is applied for every sensor of every signal in that path.
The output is the concatenated normalized fft of every sensor of every signal and the normalized frequency. 

'''



def fourier_signal_normalization(sample):

    '''
    the input is a signal and the outputs are the amplitude and the frequency of the normalized fft
    
    '''
    import numpy as np
    
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    bound = int(0.5*len(amp))
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    for i in range(0,bound):
        amp_list.append(amp[i]/max_amp)
        freq_list.append(freq[i]/max_freq)

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_vector_maker(data):

    '''The input is the concatenated signal and the outputs are two vectors, the 
    concatenated normalized amplitude and the normalized frequency'''
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_normalization(sample)[0])
        freq_vector.append(fourier_signal_normalization(sample)[1])
    return feature_vector,freq_vector


def fourier_nrm_vector(path):
    '''
    The input is the data path and the output is the concatenated normalized fft for every sensor and the normalized frequency
    
    '''
    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( fourier_vector_maker(s2)[0],fourier_vector_maker(s3)[0],fourier_vector_maker(s4)[0],fourier_vector_maker(s4)[1]),axis=1)
    return vector



############################################################

############################################################

'''

B) SIGNAL PROPERTIES EXTRACTION WITH NORMALIZATION


The following functions work together
To get the normalized fft with its properties use 'fourier_std_with_props_vector'
The input is the path and the output is the normalized fft with its properties
The properties are the excitation frequency and the reflection of the excitation frequency, their difference and the difference of their frequencies

To get the raw signal with the properties of the normalized fft use 'signal_with_props_vector' with transformation = 'none'
The input is the path and the output is the signal with the properties

To get only the properties use 'props_vector'
The input is the path and the ouptu are the properties of the normalized fft

---> signal properties extraction (signal_props_extract)
The input is a sample of normalized fft and the output is a tuple of the properties

---> signal properties extraction run (run_signal_extract)
The input is a list of signals that their fft is calculated and then are normalized after that 
for every sample the function 'signal_props_extract' is applied. The output is a list of the properties.

---> raw signal with properties (signal_with_props_vector)
The input is the path and the transformation that will be applied on the signal the output is an array of the transformed signal and the
properties of the normalized fft. It uses the function 'X_set' for the signal. The output is a concatenated array of the transformed signal and the properties
of the normalized fft.

---> properties vector (props_vector)
The input is the path and the output is an array of the concatenated properties of the normalized signal for all sensors.
For every sensor 'run_signal_extract' is applied and it calculates the properties of the signal for every sensor

---> normalized fourier signal with properties (fourier_nrm_with_props_vector)
The input is the path and the output is an array of the normalized fft and its properties
It uses the function 'X_set' for the signla which is then transformed with fft and normalized and then its properties are calculated
The output is a concatenated array of the normalized amplitude and frequency of the fft and its properties.

'''

def signal_props_extract(sample):


    '''  

    The input is a fft sample and the output is a tuple of its properties
    The bounds change depending on the material and the excitation frequency
    The default bounds are
    for the fft : freq = 0 , freq = 200 kai freq = 400 Khz
    for the normalized fft :freq = 0 , 1.3<=freq<=1.5  kai 2.9<=freq<=3.2 
    
    '''

    freq = sample[1]
    amp = sample[0]
    
    
    for i in range(0,len(freq)):

        if freq[i] >= 1.3 and freq[i] <= 1.5:
            first_bound = i
        if freq[i] >= 2.9 and freq[i] <= 3.2:
            second_bound = i
        if freq[i] ==0:
            zero_bound = i

    first_amp =[]
    for i in range(zero_bound,first_bound):
        first_amp.append(amp[i])

    second_amp =[]
    for i in range(first_bound,second_bound):
        second_amp.append(amp[i])
    
    for i in range(zero_bound,first_bound):
        if amp[i] == max(first_amp):
            first_max_amp = amp[i]
            first_max_freq = freq[i]
    

    for i in range(first_bound,second_bound):
        if amp[i] == max(second_amp):
            second_max_amp = amp[i]
            second_max_freq = freq[i]


    dx = second_max_freq-first_max_freq
    dy = first_max_amp-second_max_amp
    props = first_max_amp,second_max_amp,dx,dy
    
    return props

def run_signal_extract(data):

    '''
    The input is a list of fft samples and the output is a list of every samples' properties 
    
    '''
    feature_vector=[]
    for sample in data:
        sample = fourier_signal_normalization(sample)
        feature_vector.append(signal_props_extract(sample))
    return feature_vector

def signal_with_props_vector(path,transformation):
    

    '''
    The input is the path and the output is the transformed signal with the properties of the normalized fft

    '''

    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,freqs = X_set(path,transformation)
    vector = np.concatenate((s2,s3,s4,freqs),axis=1)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector


def props_vector(path):

    '''
    The input is the path and the output is a tuple of the normalized fft properties

    '''
    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    return vector

def fourier_nrm_with_props_vector(path):

    '''
    The input is the path and the output is the normalized fft and its properties
    
    '''
    import numpy as np
    from helper_functions import X_set
    vector = fourier_nrm_vector(path)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector
############################################################

############################################################

'''

C) HARMONICS WITH NORMALIZATION 

The following functions work together
To get the harmonics of the normalized fft use 'fourier_nrm_vector_harmonics'
That takes as input the path and the output is an array of the harmonics of every sample in that path

---> fourier harmonics (fourier_harmonics)
takes as input a signal sample and applies the fft transformation. The output is the amplitude and the frequency of the dominant harmonic.

---> harmonics from fourier signal normalization (fourier_signal_normalization_harmonics)
The input is a signal in which fft is applied and it is then normalized. Then the samples that contain the dominant harmonic are kept
and the rest is removed (These samples are 150 - 200). The output is a tuple of the normalized amplitude and frequency of the dominant harmonic. 

---> fourier harmonics vector maker (fourier_vector_maker_harmonics)
The input is an array of signal and for every signal the function 'fourier_signal_normalization_harmonics' is applied and the output is a list 
of normalized amplitudes and frequencies of the harmonic of every signal.

---> harmonics fourier nrm vector (fourier_nrm_vector_harmonics)
The input is the path and for every sample the function 'fourier_vector_maker_harmonics' is applied and the output is the concatenated array
of normalized amplitudes and frequencies of the harmonic of every sample in that path.

'''

def fourier_harmonics(sample):
    
    '''
    The input is the signal
    The output is the amplitude and the frequency of the dominant harmonic
    The dominant harmonic occurs between samples 170 and 250
    '''
    import numpy as np

    amp= fourier(sample)[0]
    freq= np.abs(fourier(sample)[1])
    amp_list =[]
    freq_list =[]

    for i in range(170,250):
        amp_list.append(amp[i])
        freq_list.append(freq[i])

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_signal_normalization_harmonics(sample):
    
    '''
    The input is a signal
    The output is the amplitude and the frequency of the normalized fft of the dominant harmonic
    '''
    
    import numpy as np
    
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    for i in range(150,200):
        amp_list.append(amp[i]/max_amp)
        freq_list.append(freq[i]/max_freq)

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_vector_maker_harmonics(data):

    '''
    The input is a list of signals
    The outputs are a list of the normalized amplitudes and a list of the normalized frequencies of the dominant harmonics
    
    '''
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_normalization_harmonics(sample)[0])
        freq_vector.append(fourier_signal_normalization_harmonics(sample)[1])
    return feature_vector,freq_vector


def fourier_nrm_vector_harmonics(path):

    '''
    The input is the path
    The output is a concatenated array of normalized amplitudes and frequencies of the harmonic of every sample in that path
    '''
    
    import numpy as np
    from helper_functions import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( fourier_vector_maker_harmonics(s2)[0],fourier_vector_maker_harmonics(s3)[0],fourier_vector_maker_harmonics(s4)[0],fourier_vector_maker_harmonics(s4)[1]),axis=1)
    return vector



############################################################


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
3) DATA TRANSFORMATIONS

---> fast fourier transform (fourier)

The input is a signal and the output is the amplitude and the frequency of the fft of the signal

---> pwelch (pwelch)
The input is a signal and the output is the amplitude and the frequency of the pwelch of the signal

---> psd (psd)
The input is a signal and the output is the amplitude and the frequency of the psd of the signal

---> spectrogram (spectrogram)
The input is a signal and the output is the spectrogram signal

---> wavelet (wavelet)
The input is a signal and the output is the db1 wavelet of the signal

---> noise adder (add_noiz)
The input is the output of the 'X_set' function and the output is the output of X_set with noise added
'''

def fourier(sample_sensor):
    '''
    The input is a signal 
    The output is the amplitude and the frequency of the fft of the signal
    '''
    import numpy as np
    fs = 1/1000
    fourier = np.fft.fft(sample_sensor)
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    power_spectrum = np.log(power_spectrum)
    
    return power_spectrum,freqs


def pwelch(sample_sensor):

    '''
    The input is a signal 
    The output is the amplitude and the frequency of the pwelch of the signal
    '''

    from scipy import signal
    fs = 1000
    (f, S)= signal.welch(sample_sensor, fs, nperseg=1024)
    return S,f
    #plt.semilogy(f, S)
    #plt.xlim([0, 500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def psd(sample_sensor):

    '''
    The input is a signal 
    The output is the amplitude and the frequency of the psd of the signal
    '''

    from scipy import signal
    fs = 1000
    (f, S) = signal.periodogram(sample_sensor, fs, scaling='density')
    return S,f
    #plt.semilogy(f, S)
    #plt.ylim([1e-14, 1e-3])
    #plt.xlim([0,500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def spectrogram(sample):

    '''
    The input is a signal 
    The output is the spectrogram of the signal
    '''

    from scipy import signal

    fs = 1000
    f, t, Sxx = signal.spectrogram(sample, fs)
    #plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    return Sxx

def wavelet(sample):
    '''
    The input is a signal 
    The output is the db1 wavelet of the signal
    '''
    import pywt
    import numpy as np

    fs = 1000  
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = sample

    wavelet_name = 'db1' 
    transformed_signal, _ = pywt.dwt(signal, wavelet_name)
    return transformed_signal
    # Plot the original signal
    #plt.subplot(2, 1, 1)
    #plt.plot(signal)
    #plt.title('Original Signal')

    # Plot the transformed signal
    #plt.subplot(2, 1, 2)
    #plt.plot(transformed_signal)
    #plt.title('Transformed Signal')

    #plt.tight_layout()
    #plt.show()

def add_noiz(X, noise_percent):
    import numpy as np
    '''
    Adds Gaussian noise to X.
    Handles both 2D numpy arrays and lists of 1D arrays (e.g. from X_set with 'fourier' or 'none').
    '''
    if isinstance(X, np.ndarray):
        # Case 1: 2D numpy array (samples x features)
        std_dev = np.std(X, axis=0)
        noise = np.random.randn(*X.shape) * (noise_percent / 100.0) * std_dev
        return X + noise

    elif isinstance(X, list) or isinstance(X, tuple):
        X_noisy = []
        for sample in X:
            sample = np.asarray(sample)
            std_dev = np.std(sample)
            noise = np.random.randn(*sample.shape) * (noise_percent / 100.0) * std_dev
            noisy_sample = sample + noise
            X_noisy.append(noisy_sample)
        return X_noisy

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


'''
4) FEATURE ENGINEERING TECHNIQUES

---> prinicipal component analysis (pca)
The input is the X_train and X_test and this function performs pca. PCA finds linear combinations with the largest variance and creates principal 
components with these combinations. The results are the X_train and X_test that instead of the original data they contain the principal components.

---> kernel principal component analysis (kpca)
The input is the X_train and X_test and this function performs kernel pca. First the data are projected in a feature space according to the choice of the kernel. 
Then PCA finds linear combinations with the largest variance and creates principal components with these combinations. The results are the X_train and X_test that 
instead of the original data they contain the principal components from the feature space they were projected.

---> data mixer (data_mixer)
The input is the X and Y data of one dataset and those of another dataset and the percentage that is kept from each dataset
Then the percentages of each dataset are combined and shuffled
The output is the shuffled combination of the two datasets.
'''


def pca(X_train,X_test):

    '''
    The input is the original X_train and X_test
    The output is the X_train and X_test that contain the principal components of the original data
    '''
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    pca = PCA(n_components=50, random_state = 42)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pca.transform(X_test)
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    #plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    #plt.ylabel('Percentage of Explained Variance')
    #plt.xlabel('Principal Component')
    #plt.title('Scree Plot')
    #plt.show()
    return X_train,X_test


def kpca(X_train,X_test,input_kernel):


    '''
    The input is the original X_train and X_test and the kernel
    The output is the X_train and X_test that contain the principal components of the original data that are projected to a feature space 
    The feature space depends on the chosen kernel

    The kernels are :
    Periodic
    Locally periodic
    RBF
    Rational quadratic
    Rational locally periodic
    '''
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import KernelPCA   
    from sklearn.gaussian_process.kernels import ExpSineSquared,Product,RationalQuadratic,RBF

    periodic= ExpSineSquared()
    locally_periodic = Product(periodic,RBF())
    rational_locally_periodic = Product(periodic,RationalQuadratic())


    if input_kernel =='periodic':
        input_kernel = periodic
    if input_kernel =='locally_periodic':
        input_kernel = locally_periodic
    if input_kernel =='rbf':
        input_kernel = RBF()
    if input_kernel =='rational_quadratic':
        input_kernel = RationalQuadratic()
    if input_kernel =='rational_locally_periodic':
        input_kernel = rational_locally_periodic
    
    pca = KernelPCA(kernel=input_kernel,n_components=30, random_state = 42)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pca.transform(X_test)
    X_test = pd.DataFrame(X_test)
    return X_train,X_test

def data_mixer(X_1,y_1,X_2,y_2,first_percentage,second_percentage):
    '''
    
    The input is the X and Y data of two datasets each dataset's percentage that is kept
    The output is the shuffled combination of the two datasets.
    
    '''
    from sklearn.model_selection import train_test_split
    import numpy as np
    if first_percentage == 1:
        X_1_half = X_1
        y_1_half = y_1        
    else:
        X_1_half, X_drop, y_1_half, y_drop = train_test_split(X_1, y_1, test_size=1-first_percentage,shuffle=True)
    
    if second_percentage ==1:
        X_2_half = X_2
        y_2_half = y_2
    else:
        X_2_half, X_drop, y_2_half, y_drop = train_test_split(X_2, y_2, test_size=1-second_percentage,shuffle=True)
    
    X_train = np.concatenate((X_1_half,X_2_half),axis=0)
    y_train = np.concatenate((y_1_half,y_2_half),axis=0)
    return X_train,y_train


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
5) PLOTS



---> figures with subplots of all damage scenarios for every sensor(all_damage_every_sensor_separate)
takes as input the data path and a list that contains tuples which contain the indexes of each sample, the damage 
percentage values or defects, the color of the line, and the linestyle. It plots three figures, one for every sensor,
each figure contains subplots of all damage scenarios of the given samples.

---> figure of all damage scenarios for every sensor(all_damage_every_sensor_together)
takes as input the data path and a list that contains tuples which contain the indexes of each sample, the damage 
percentage values or defects, the color of the line, and the linestyle. It plots three figures, one for every sensor,
each figure contains plots of all damage scenarios of the given samples.

---> signal plot of every sensor (all_sensor_time_plot_separate)
takes as input the data path and the index of the sample that is plotted. Plots three subplots of the 
time signal of every sensor for the sample of that index.

---> fft plot of every sensor (all_sensor_fft_plot_separate)
takes as input the data path and the index of the sample that is plotted. Plots three subplots of the 
fft of every sensor for the sample of that index.

---> fft plot of a single sensor (single_sensor_fft_plot)
takes as input the data path the index of the sample that is plotted and the name of the defect and plots the fft of sensor 3. 
Two arrows show the excitation frequency and the dominant harmonic frequency.

---> harmonics plot for every defect mode (every_defect_mode_harmonics_plot)
takes as input the data path and the sample indexes for every kind of defect(dd,df,dm,all) and plots 4
subplots of the harmonics of every defect mode for one sensor

---> regression results bar chart (regression_results_bar_charts)
the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists, the label on y axis, the noise level and the number of datapoints.

---> classification results bar chart (classification_results_bar_charts)
the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists, the label on y axis, the noise level and the number of datapoints.

---> parity plots (parity_plot_from_csv)
the inputs are the csv containing the results and the mode 'save' or 'show' plots are made for all results in that csv file
In 'save' mode the plot is saved and in 'show' mode the plot is shown


---> confusion matrices (confusion_matrix_display_from_csv)
the inputs are the csv containing the results and the mode 'save' or 'show' plots are made for all results in that csv file
In 'save' mode the plot is saved and in 'show' mode the plot is shown

---> plot signal and fft with noise(plot_signal_with_variants)
the input is the csv path of the sample and the output is a figure with two subpplots each containing the plots of signal and the fft with their three noise levels 
'''


def all_damage_every_sensor_separate(path,index_list):
    
    '''
    takes as input the data path and a list that contains tuples which contain the indexes of each sample, the damage 
    percentage values or defects, the color of the line, and the linestyle
    The output are three figures, one for every sensor,
    each figure contains subplots of all damage scenarios of the given samples.
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    from helper_functions import fourier, X_set

    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.linewidth'] = 2.5

    __, s2, s3, s4, __ = X_set(path, 'none')
    sensor_list = [(s2, 'sensor 2'), (s3, 'sensor 3'), (s4, 'sensor 4')]

    for sensor_data, sensor_name in sensor_list:
        num_plots = len(index_list)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2.8 * num_plots), sharex=True)

        if num_plots == 1:
            axs = [axs]

        for ax, (idx, label, color, linestyle) in zip(axs, index_list):
            fft_amplitude, fft_freq = fourier(sensor_data[idx])
            ax.plot(np.abs(fft_freq), fft_amplitude,
                    color=color, linestyle=linestyle, label=label)
            ax.grid(True)
            ax.legend(loc='upper right')

        axs[-1].set_xlabel('Frequency (kHz)')

        fig.text(0.04, 0.5, 'Log Amplitude (V)', va='center', rotation='vertical', fontsize=16)
        fig.suptitle(f'FFT plots of {sensor_name}', fontsize=18)
        fig.tight_layout(rect=[0.06, 0, 1, 0.95])
        plt.show()


def all_damage_every_sensor_together(path,index_list):
    
    '''
    The input is the data path and a list that contains tuples which contain the indexes of each sample, the damage 
    percentage values or defects, the color of the line, and the linestyle.
    The outputs are three figures, one for every sensor,
    each figure contains plots of all damage scenarios of the given samples.
    '''

    import matplotlib.pyplot as plt
    import numpy as np
    from helper_functions import fourier, X_set

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    __, s2, s3, s4, __ = X_set(path, 'none')
    sensor_list = [(s2, 'sensor 2'), (s3, 'sensor 3'), (s4, 'sensor 4')]

    for sensor_data, sensor_name in sensor_list:
        plt.figure()
        legend_list = []
        y_offset = 0 

        for idx, damage, linecolor, linestyle in index_list:
            fft_amplitude, fft_freq = fourier(sensor_data[idx])
            
            amplitude_with_offset = fft_amplitude + y_offset
            plt.plot(np.abs(fft_freq), amplitude_with_offset, color=linecolor, linestyle=linestyle)
            legend_list.append(f"{damage} (offset {y_offset})")
            y_offset += 2.5

        plt.title(f'FFT in log scale of {sensor_name}')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Log Amplitude + Offset')
        plt.legend(legend_list)
        plt.grid(True)
        plt.show()

def all_sensor_time_plot_separate(path,index):

    '''
    The input is the data path and the index of the sample that is plotted. 
    The outputs are three subplots of the time signal of every sensor for the sample of that index.
    '''
    
    import matplotlib.pyplot as plt

    __,s2,s3,s4,__ = X_set(path,'none')

    s2 = s2[index]
    s3 = s3[index]
    s4 = s4[index]

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    fig,axs = plt.subplots(3,sharex=True,sharey=True)
    fig.suptitle('Signal from all sensors (all defect modes)')
    fig.text(0.5 , 0.04, 'time (ms)',ha = 'center')
    fig.text(0.04 , 0.5, 'Electric potential (v)',va = 'center',rotation = 'vertical')

    axs[0].plot(s2)
    axs[0].grid()
    axs[0].set_title('s2')

    axs[1].plot(s3)
    axs[1].grid()
    axs[1].set_title('s3')

    axs[2].plot(s4)
    axs[2].grid()
    axs[2].set_title('s4')

    plt.show()


def all_sensor_fft_plot_separate(path,index):


    '''
    The input is the data path and the index of the sample that is plotted.
    The outputs are three subplots of the fft of every sensor for the sample of that index
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    __,s2,s3,s4,__ = X_set(path,'none')

    s2 = fourier(s2[index])
    s3 = fourier(s3[index])
    s4 = fourier(s4[index])

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    fig,axs = plt.subplots(3,sharex=True,sharey=True)
    fig.suptitle('Signal with FFT transformation in log scale (all defect modes)')
    fig.text(0.5 , 0.04, 'Frequency (kHz)',ha = 'center')
    fig.text(0.04 , 0.5, 'Amplitude (v)',va = 'center',rotation = 'vertical')

    axs[0].plot(np.abs(s2[1]),s2[0])
    axs[0].grid()
    axs[0].set_title('s2')

    axs[1].plot(np.abs(s3[1]),s3[0])
    axs[1].grid()
    axs[1].set_title('s3')

    axs[2].plot(np.abs(s4[1]),s4[0])
    axs[2].grid()
    axs[2].set_title('s4')

    plt.show()


def single_sensor_fft_plot(path,index,defect):
    
    '''
    The input is the data path, the index of the sample that is plotted and the name of the defect 
    The output is a plot of the fft of sensor 3. 
    Two arrows show the excitation frequency and the dominant harmonic frequency.
    
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    __,s2,s3,s4,__ = X_set(path,'none')

    s2 = fourier(s2[index])
    s3 = fourier(s3[index])
    s4 = fourier(s4[index])

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    
    plt.title(f'Signal with FFT transformation in log scale of sensor 3 ({defect})')
    plt.xlabel( 'Frequency (kHz)')
    plt.ylabel('Amplitude (v)')
    plt.plot(np.abs(s3[1]),s3[0])
    plt.grid()
    plt.annotate('Excitation frequency (125kHz)', xy=(125,-4), xytext=(200, -3),
    arrowprops=dict(facecolor='black', shrink=0.2))
    plt.annotate('Dominant harmonic (250 kHz)', xy=(250,-10), xytext=(120,-12),
    arrowprops=dict(facecolor='black', shrink=0.2))
    plt.show()

def every_defect_mode_harmonics_plot(path,dd_index,df_index,all_index,dm_index):

    '''
    The input is the data path and the sample indexes for every kind of defect(dd,df,dm,all) 
    The outputs are 4 subplots of the harmonics of every defect mode for one sensor

    to see all defects:
    path = 'Balanced_data'

    dd --> 0
    df --> 1
    all --> 54
    dm --> 58
    '''

    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5
    __,s2,s3,s4,__ = X_set(path,'none')

    dd = fourier_harmonics(s3[dd_index])
    df = fourier_harmonics(s3[df_index])
    all = fourier_harmonics(s3[all_index])
    dm = fourier_harmonics(s3[dm_index])
    fig,axs = plt.subplots(4,sharex=True,sharey=True)


    fig.suptitle(f'Harmonics in log scale of sensor 3')
    fig.text(0.5 , 0.04, 'Frequency (kHz)',ha = 'center')
    fig.text(0.04 , 0.5, 'Amplitude (v)',va = 'center',rotation = 'vertical')

    axs[0].plot(df[1],df[0])
    axs[0].grid()
    axs[0].set_title('Fiber failure')

    axs[1].plot(dm[1],dm[0])
    axs[1].grid()
    axs[1].set_title('Matrix failure')

    axs[2].plot(dd[1],dd[0])
    axs[2].grid()
    axs[2].set_title('Delamination')

    axs[3].plot(all[1],all[0])
    axs[3].grid()
    axs[3].set_title('All defect modes')


    plt.show()


def regression_results_bar_charts(model_names, mape, std_devs, pvals, ylabel, noise, n_points, data_percentage):
    '''
    the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists,
    the label on y axis, the noise level, the number of datapoints, and the data percentage.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    # Map number of points to frequency labels
    point_freq_map = {
        375: "250 kHz",
        460: "300 kHz",
        750: "500 kHz"
    }
    freq_label = point_freq_map.get(n_points, f"{n_points} points")

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    mape = np.clip(np.array(mape), 0.001, None)
    std_devs = np.array(std_devs)
    pvals = np.array(pvals)

    model_count = len(model_names)
    x = np.arange(model_count)
    bar_width = 0.45

    fig, ax = plt.subplots(figsize=(20, 10))

    bars_base = ax.bar(x - bar_width / 2, mape[:, 0], width=bar_width, label='Time',
                       color='skyblue', edgecolor='black')
    bars_fourier = ax.bar(x + bar_width / 2, mape[:, 1], width=bar_width, label='fourier',
                          color='salmon', edgecolor='black')

    for i in range(model_count):
        for j, bar_set in enumerate([bars_base, bars_fourier]):
            height = bar_set[i].get_height()
            std = std_devs[i][j]
            pval = pvals[i][j]
            label = f'{mape[i][j]:.4f} % ± {std:.4f} %\n(p: {pval:.2e})'
            ax.text(bar_set[i].get_x() + bar_set[i].get_width() / 2,
                    height * 1.1,
                    label,
                    ha='center', va='bottom', fontsize=9)

    ax.set_yscale('log')
    yticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_yticks([y for y in yticks if y <= mape.max() * 1.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f} %'))

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    title = f'Regression Model Performance (Noise: {noise}%, {freq_label}, Data: {data_percentage})'
    ax.set_title(title)
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'results/reg{noise}{n_points}{data_percentage}.png', bbox_inches='tight', dpi=300)
    plt.close()


def class_results_bar_charts(model_names, mape, std_devs, pvals, ylabel, noise, n_points, data_percentage):
    '''
    the inputs are the model names, the mape values, the standard deviation values, the p-value values as lists,
    the label on y axis, the noise level, the number of datapoints, and the data percentage.
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    # Map number of points to frequency labels
    point_freq_map = {
        375: "250 kHz",
        460: "300 kHz",
        750: "500 kHz"
    }
    freq_label = point_freq_map.get(n_points, f"{n_points} points")

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['lines.linewidth'] = 2.5

    mape = np.array(mape)
    std_devs = np.array(std_devs)
    pvals = np.array(pvals)

    model_count = len(model_names)
    x = np.arange(model_count)
    bar_width = 0.45

    fig, ax = plt.subplots(figsize=(20, 10))

    bars_base = ax.bar(x - bar_width/2, mape[:, 0], width=bar_width, label='Time',
                       color='skyblue', edgecolor='black')
    bars_fourier = ax.bar(x + bar_width/2, mape[:, 1], width=bar_width, label='fourier',
                          color='salmon', edgecolor='black')

    for i in range(model_count):
        for j, bar_set in enumerate([bars_base, bars_fourier]):
            height = bar_set[i].get_height()
            std = std_devs[i][j]
            pval = pvals[i][j]
            label = f'{mape[i][j]:.4f} ± {std:.4f} \n(F1 macro: {pval:.4f})'
            ax.text(bar_set[i].get_x() + bar_set[i].get_width() / 2,
                    height + 0.01,
                    label,
                    ha='center', va='bottom', fontsize=10.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    title = f'Classification Model Performance (Noise: {noise}%, {freq_label}, Data: {data_percentage})'
    ax.set_title(title)
    ax.set_ylim(0, np.max(mape + std_devs) + 0.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'results/clf{noise}{n_points}{data_percentage}.png', bbox_inches='tight', dpi=300)
    plt.close()



def parity_plot_from_csv(csv_path, mode='show', font_scale=1.5):
    '''
    The input is the csv path that contains the results from all the experiments
    and the mode 'save' or 'show'
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import ast
    import os

    # Font sizes
    title_size = 12 * font_scale
    axis_title_size = 12 * font_scale
    tick_label_size = 10 * font_scale
    legend_size = 10 * font_scale

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        model = row['model']
        transformation = row['transformation']
        noise = row['noise_percent']
        n_points = row['n_points']
        mean_mape = row['mean_mape']
        std_mape = row['std_mape']
        pval = row['pval']

        # Modify the unit for n_points based on transformation
        if transformation == 'none':
            n_points_str = f'{n_points} μs'
        elif transformation == 'fourier':
            n_points_str = f'{n_points} kHz'
        else:
            n_points_str = str(n_points)

        y_test = np.array(ast.literal_eval(row['last_fold_true']))
        y_pred = np.array(ast.literal_eval(row['last_fold_preds']))

        plt.figure(figsize=(6, 6))  # Square figure
        plt.scatter(y_test, y_pred, color='r', label='Predicted vs True', s=30)
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='y = x')

        plt.xlabel('True Values', fontsize=axis_title_size)
        plt.ylabel('Predicted Values', fontsize=axis_title_size)
        plt.title(
            f'Parity Plot - {model} | Noise: {noise}% | Transform: {transformation} | N={n_points_str}',
            fontsize=title_size
        )
        plt.xticks(fontsize=tick_label_size)
        plt.yticks(fontsize=tick_label_size)

        # Force square aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')

        legend_text = (
            f'MAPE: {100*mean_mape:.6f}% ± {100*std_mape:.6f}%\n'
            f'P-value: {pval:.2e}'
        )
        plt.legend(title=legend_text, loc='upper left', fontsize=legend_size, title_fontsize=legend_size)

        if mode == 'save':
            filename = f'{model}_parity_n{n_points}_noise{noise}_transf_{transformation}.png'.replace(" ", "_")
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


def confusion_matrix_display_from_csv(csv_path, mode='show', font_scale=1.5):
    '''
    The input is the csv path that contains the results from all the experiments
    and the mode 'save' or 'show'
    '''
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import ast
    import os

    # Custom label mapping
    label_mapping = {
        0: "all defect modes",
        1: "dd",
        2: "df",
        3: "dm"
    }
    class_labels = [label_mapping[i] for i in sorted(label_mapping.keys())]

    # Font sizes
    title_size = 14 * font_scale
    axis_title_size = 12 * font_scale
    tick_label_size = 10 * font_scale
    legend_size = 10 * font_scale

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        model = row['model']
        transformation = row['transformation']
        noise = row['noise_percent']
        n_points = row['n_points']
        mean_acc = row['mean_acc']
        std_acc = row['std_acc']
        f1 = row['f1_macro']

        # Modify the unit for n_points based on transformation
        if transformation == 'none':
            n_points_str = f'{n_points} μs'
        elif transformation == 'fourier':
            n_points_str = f'{n_points} kHz'
        else:
            n_points_str = str(n_points)

        # Load predicted and true labels
        y_test = np.array(ast.literal_eval(row['last_fold_true']))
        y_pred = np.array(ast.literal_eval(row['last_fold_preds']))

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=sorted(label_mapping.keys()))

        # Display with class names
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, colorbar=False)

        # Increase axis label font sizes
        ax.set_xlabel("Predicted label", fontsize=axis_title_size)
        ax.set_ylabel("True label", fontsize=axis_title_size)

        # Increase tick label font size
        ax.tick_params(axis='both', labelsize=tick_label_size)

        plt.title(
            f'Confusion Matrix - {model} | Noise: {noise}% | Transform: {transformation} | N={n_points_str}',
            fontsize=title_size
        )

        legend_text = (
            f'Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n'
            f'F1 Macro: {f1:.2f}'
        )
        plt.text(
            1.05, 0.95, legend_text,
            transform=ax.transAxes,
            fontsize=legend_size,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9)
        )

        if mode == 'save':
            filename = f'{model}_conf_matrix_n{n_points}_noise{noise}_transf_{transformation}.png'.replace(" ", "_")
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()



def plot_signal_with_variants(csv_path: str,
                              signal_column: str = "s4",
                              time_column: str = "time",
                              noise_levels=(2, 5, 10),
                              alpha_levels=(0.6, 0.4, 0.2),
                              linewidths=(3.0, 3.5, 4.0),
                              time_color: str = "blue",
                              fft_color: str = "orange",
                              save: bool = False,
                              show: bool = True,
                              output_filename: str = None,
                              seed: int = 42):
    
    '''
    
    the input is the csv path of the sample 
    
    '''

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")


    

    df = pd.read_csv(csv_path, sep=r"\s+")

    if signal_column not in df.columns:
        raise ValueError(f"Column '{signal_column}' not found. Available: {list(df.columns)}")
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found. Available: {list(df.columns)}")

    time = df[time_column].values
    signal = df[signal_column].values
    np.random.seed(seed)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # --- TIME DOMAIN ---
    axes[0].plot(time, signal, color=time_color, linewidth=2.0, label="Original Signal", zorder=10)
    for nl, alpha, lw in zip(noise_levels, alpha_levels, linewidths):
        noisy_signal = add_noiz(signal, nl)
        axes[0].plot(time, noisy_signal, color=time_color, alpha=alpha, linewidth=lw, label=f"Noise {nl}%", zorder=5)

    axes[0].set_xlabel("Time (ms)", fontsize=12)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].set_title(f"Time-Domain Signal of {signal_column}", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # --- FFT (same as your original script) ---
    power_orig, freqs = fourier(signal)
    mask = freqs >= 0  # Keep only positive frequencies
    freqs = freqs[mask]
    power_orig = power_orig[mask]
    axes[1].plot(freqs, power_orig, color=fft_color, linewidth=2.0, label="Original FFT", zorder=10)

    for nl, alpha, lw in zip(noise_levels, alpha_levels, linewidths):
        noisy_signal = add_noiz(signal, nl)
        power_noisy, freqs_noisy = fourier(noisy_signal)
        axes[1].plot(freqs_noisy[mask], power_noisy[mask], color=fft_color, alpha=alpha, linewidth=lw, label=f"Noise {nl}%", zorder=5)

    axes[1].set_xlabel("Frequency (kHz)", fontsize=12)
    axes[1].set_ylabel("Log Amplitude", fontsize=12)
    axes[1].set_title(f"FFT of {signal_column} ", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save:
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            output_filename = f"{base_name}_{signal_column}_signal_fft_noise.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Figure saved to: {output_filename}")

    if show:
        plt.show()
    else:
        plt.close()


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################



'''

6)EXPERIMENT RUN

---> regression experiment run (regression_model_run)
The input is the model as a function the X_train, y_train, X_test, y_test. That model is trained and tested and mae and mape are calculated.
The outputs are mae,mape,y_test and y_pred 

---> classification experiment run(classification_model_run)
The input is the model as a function the X_train, y_train, X_test, y_test. That model is trained and tested and its accuracy is calculated.
The outputs are accuracy,y_test kai y_pred
'''


def regression_model_run(model,X_train,y,X_test,y_test):
    
    '''
    The input is the model as a function the X_train, y_train, X_test, y_test.
    The outputs are mae,mape,y_test and y_pred of that model
    '''
    from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error

    y_pred = model(X_train,y,X_test)
    mape = 100*mean_absolute_percentage_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    return mae,mape,y_test,y_pred

def classification_model_run(model,X_train,y,X_test,y_test):

    '''
    The input is the model as a function the X_train, y_train, X_test, y_test.
    The outputs are accuracy,y_test and y_pred of that model
    '''

    from sklearn.metrics import accuracy_score
    y_pred = model(X_train,y,X_test)
    acc = 100*accuracy_score(y_test,y_pred)
    acc = accuracy_score(y_test,y_pred)
    return acc,y_test,y_pred

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################



'''

7)TOOLS FOR TUNING

---> cross validation me leave one out(cross_val_loo)

The input is the model, the X and y data. This function performs leave one out cross validation and calculates the scores
for every fold. For regression the scoring is 'neg_mean_absolute_percentage_error'  and for classification the scoring is 'accuracy'
The output is the score of each fold


---> grid search me leave one out(grid_search_loo)
The input is the model the X_train and y_train. This function performs grid search with leave one out to find the best parameters for the model according to a scoring.
Depending on the model the parameters defer.
The outputs are the parameters with which the model achieved its best performance
'''

def cross_val_loo(model,X,y):
    '''
    The input is the model, the X and y data.
    The output is the score of each fold after cross validation with leave one out
    
    
    regression --> scoring = 'neg_mean_absolute_percentage_error'
    classification ---> scoring = 'accuracy'
    '''
    from sklearn.model_selection import LeaveOneOut,cross_val_score
    import numpy as np
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return np.absolute(scores)



def grid_search_loo(model,X_train,y_train):
    '''
    The input is the model the X_train and y_train
    The outputs are the parameters with which the model achieved its best performance after leave one out grid search

    Each model has different parameters, the default parameters are the parameters of an SVM algorithm
    
    '''
    from sklearn.model_selection import GridSearchCV,LeaveOneOut
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf','sigmoid','poly']} 

    grid = GridSearchCV(model, param_grid, refit = True, cv=LeaveOneOut(),verbose = False) 
    grid.fit(X_train, y_train) 

    print(grid.best_params_)
    print(grid.best_score_)


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################





'''

8)EXPERIMENTAL RESULTS EXTRACTION

---> Regression data extraction (extract_regression_data)
Extracts data from regression experiments, specifically for bar chart function: regression_results_bar_charts



---> Classification data extraction (extract_classification_data)
Extracts data from Classification experiments, specifically for bar chart function: class_results_bar_charts


---> Process results in csv form and plot (process_plot_csv_files)
Opens the results for both classification and regression and provides the bar charts using the functions:extract_regression_data,regression_results_bar_charts
and extract_classification_data,class_results_bar_charts

'''

def extract_regression_data(csv_path):
    import numpy as np
    import pandas as pd
    """
    Extracts model names, MAPE values, standard deviations, and p-values
    from a regression results CSV. Also returns noise level, number of points, and data percentage.
    """
    df = pd.read_csv(csv_path)

    # Get metadata
    n_points = df['n_points'].iloc[0]
    noise = df['noise_percent'].iloc[0]
    data_percentage = df['data_percentage'].iloc[0] if 'data_percentage' in df.columns else 100

    assert {'none', 'fourier'}.issubset(set(df['transformation'])), \
        "Missing expected transformations: 'none' and 'fourier'"

    base_models = df['model'].unique()
    model_names = []
    mape = []
    std_devs = []
    pvals = []

    for model in base_models:
        base_row = df[(df['model'] == model) & (df['transformation'] == 'none')].iloc[0]
        fourier_row = df[(df['model'] == model) & (df['transformation'] == 'fourier')].iloc[0]

        model_names.append(model)
        mape.append([base_row['mean_mape'] * 100, fourier_row['mean_mape'] * 100])
        std_devs.append([base_row['std_mape'] * 100, fourier_row['std_mape'] * 100])
        pvals.append([base_row['pval'], fourier_row['pval']])

    return model_names, np.array(mape), np.array(std_devs), np.array(pvals), noise, n_points, data_percentage


def extract_classification_data(csv_path):
    import pandas as pd
    import numpy as np
    """
    Extracts model names, accuracy values, standard deviations, and F1 scores
    from a classification results CSV. Also returns noise level, number of points, and data percentage.
    """
    df = pd.read_csv(csv_path)

    n_points = df['n_points'].iloc[0]
    noise = df['noise_percent'].iloc[0]
    data_percentage = df['data_percentage'].iloc[0] if 'data_percentage' in df.columns else 100

    assert {'none', 'fourier'}.issubset(set(df['transformation'])), \
        "Missing expected transformations: 'none' and 'fourier'"

    base_models = df['model'].unique()
    model_names = []
    acc = []
    std_devs = []
    f1_scores = []

    for model in base_models:
        base_row = df[(df['model'] == model) & (df['transformation'] == 'none')].iloc[0]
        fourier_row = df[(df['model'] == model) & (df['transformation'] == 'fourier')].iloc[0]

        model_names.append(model)
        acc.append([base_row['mean_acc'], fourier_row['mean_acc']])
        std_devs.append([base_row['std_acc'], fourier_row['std_acc']])
        f1_scores.append([base_row['f1_macro'], fourier_row['f1_macro']])

    return model_names, np.array(acc), np.array(std_devs), np.array(f1_scores), noise, n_points, data_percentage


def process_plot_csv_files(folder_path):
    import os
    import pandas as pd
    """
    Processes CSV files in the given folder:
    - Splits each file by unique noise_percent and data_percentage values.
    - For classification files, runs class_results_bar_charts on each subset.
    - For regression files, runs regression_results_bar_charts on each subset.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if "classification" in filename.lower():
                for noise in sorted(df['noise_percent'].unique()):
                    for perc in sorted(df['data_percentage'].unique()):
                        subset = df[(df['noise_percent'] == noise) & (df['data_percentage'] == perc)]
                        temp_path = os.path.join(folder_path, f"temp_classification_noise_{noise}_perc_{perc}.csv")
                        subset.to_csv(temp_path, index=False)

                        model_names, acc, stds, f1_macro, _, n_points, data_percentage = extract_classification_data(temp_path)
                        class_results_bar_charts(model_names, acc, stds, f1_macro,
                                                 ylabel='Accuracy',
                                                 noise=noise,
                                                 n_points=n_points,
                                                 data_percentage=data_percentage)

                        os.remove(temp_path)

            elif "regression" in filename.lower():
                for noise in sorted(df['noise_percent'].unique()):
                    for perc in sorted(df['data_percentage'].unique()):
                        subset = df[(df['noise_percent'] == noise) & (df['data_percentage'] == perc)]
                        temp_path = os.path.join(folder_path, f"temp_regression_noise_{noise}_perc_{perc}.csv")
                        subset.to_csv(temp_path, index=False)

                        model_names, mape, stds, pvals, _, n_points, data_percentage = extract_regression_data(temp_path)
                        regression_results_bar_charts(model_names, mape, stds, pvals,
                                                      ylabel='MAPE (%)',
                                                      noise=noise,
                                                      n_points=n_points,
                                                      data_percentage=data_percentage)

                        os.remove(temp_path)


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

