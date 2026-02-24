conda create --name eeg-ssl python==3.9
source activate base
conda activate eeg-ssl

conda install -c conda-forge datalad
pip install -r requirements.txt

pip install git+https://github.com/SPOClab-ca/dn3#egg=dn3
