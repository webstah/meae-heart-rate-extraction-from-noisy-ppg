import pyedflib
from glob import glob
import numpy as np
from scipy.signal import resample, resample_poly
from multiprocessing import Process
import pickle

def generate_mesa_data(psg_file_paths, start, end):
    
    # count = 0
    for subject_num in range(start, end):
        print('-------------------------------------------------------------------------')
        print('Preprocessing training subject: %d/%d' % (subject_num + 1, np.size(psg_file_paths, 0)))
        print(psg_file_paths[subject_num])

        record_name = psg_file_paths[subject_num][:-4]
        for name_index in range(0, -40, -1):
            if record_name[name_index] == '\\':
                subject_name = record_name[name_index + 1:]
                break
            
        # load PSG data
        f = pyedflib.EdfReader(psg_file_paths[subject_num])
        signal_names = f.getSignalLabels()
        
        ppg_fs = 0
        # extract ECG and PPG signals - the name checks are in place due to legacy code, but should be kept in place
        for signal_num in range(len(signal_names)):
            if signal_names[signal_num][:3] == "PPG" or signal_names[signal_num].find('ppg') > 0 or signal_names[signal_num].find('Pleth') > 0 or signal_names[signal_num] == "PLETH" or signal_names[signal_num] == "Pleth":
                ppg_signal_num = signal_num
                ppg_fs = f.getSampleFrequency(signal_num)
            if signal_names[signal_num][:3] == "ECG" or signal_names[signal_num][:3] == "EKG" or signal_names[signal_num][:3] == "ecg" or signal_names[signal_num][:3] == "ekg" or signal_names[signal_num][:5] == "Heart":
                if signal_names[signal_num][-1] != 'F' and f.getSampleFrequency(signal_num) > 40:
                    ecg_signal_num = signal_num
                    ecg_fs = f.getSampleFrequency(signal_num)

        if ppg_fs ==0:
            print("one of the signals doesnt exist")
        else:
            ecg_signal = f.readSignal(ecg_signal_num)
            print('signal name: ' + signal_names[ecg_signal_num] + ', fs: ' + str(ecg_fs), ', len: ' + str(len(ecg_signal)))
            ecg_signal = resample_poly(ecg_signal, up=125, down=ppg_fs)
            ppg_signal = f.readSignal(ppg_signal_num)
            print('signal name: ' + signal_names[ppg_signal_num] + ', fs: ' + str(ppg_fs), ', len: ' + str(len(ppg_signal)))
            ppg_signal = resample_poly(ppg_signal, up=125, down=ppg_fs)
            
            data = np.vstack((ecg_signal, ppg_signal))
            print(np.shape(data))
            filename = './data/mesa_ppg_ecg/' + subject_name + '.dat'
            with open(filename, 'wb') as file:
                pickle.dump(data, file)

        print('-------------------------------------------------------------------------')


if __name__ == '__main__':
    DATA_PATH = 'Z:/data/Polysomnography/mesa/polysomnography/edfs/*.edf'

    # change the datapath to your datapath containing .edf files
    psg_file_paths = glob(DATA_PATH)
    num_workers = 12
    
    num_to_process_per_worker = int(len(psg_file_paths) / num_workers)
    left_over = len(psg_file_paths)-num_workers*num_to_process_per_worker
    for i in range(num_workers):
        start = i*num_to_process_per_worker
        end = (i+1)*num_to_process_per_worker
        if i == num_workers-1:
            end += left_over
        worker = Process(target=generate_mesa_data, args=(psg_file_paths, start, end))
        print('starting id ', str(i), 'with', str(end-start), 'number of subjects')
        worker.start()