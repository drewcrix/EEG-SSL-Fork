wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/

kaggle datasets download esantamaria/gibuva-erpbci-dataset
kaggle competitions download -c inria-bci-challenge