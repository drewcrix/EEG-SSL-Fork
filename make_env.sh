conda create --name bendr_env python=3.9 -y
conda activate bendr_env

conda install -c conda-forge git-annex=*=alldep*                           


# install torch first — pick the line that matches your machine:
# GPU (CUDA 12.6):  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# GPU (CUDA 11.8):  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:         pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

pip install git+https://github.com/SPOClab-ca/dn3#egg=dn3

pip install pytest

# patch 1: remove nntplib import removed in Python 3.13
python -c "
import os, dn3
f = os.path.join(os.path.dirname(dn3.__file__), 'trainable', 'models.py')
t = open(f, encoding='utf-8').read().replace('import nntplib\n', '')
open(f, 'w', encoding='utf-8').write(t)
print('patched dn3 models.py')
"

# patch 2: fix collections.Iterable removed in Python 3.10+
python -c "
import os, dn3
f = os.path.join(os.path.dirname(dn3.__file__), 'utils.py')
t = open(f, encoding='utf-8').read().replace('from collections import Iterable', 'from collections.abc import Iterable')
open(f, 'w', encoding='utf-8').write(t)
print('patched dn3 utils.py')
"

# patch 3: fix moabb using np.int/np.bool/np.object removed in numpy 1.24+
python -c "
import os, moabb
f = os.path.join(os.path.dirname(moabb.__file__), 'datasets', 'neiry.py')
t = open(f, encoding='utf-8', errors='replace').read()
t = t.replace('(\"id\", np.int)', '(\"id\", np.int64)')
t = t.replace('(\"target\", np.int)', '(\"target\", np.int64)')
t = t.replace('(\"is_train\", np.bool)', '(\"is_train\", np.bool_)')
t = t.replace('(\"prediction\", np.int)', '(\"prediction\", np.int64)')
t = t.replace('(\"sessions\", np.object)', '(\"sessions\", object)')
t = t.replace('(\"eeg\", np.object)', '(\"eeg\", object)')
t = t.replace('(\"starts\", np.object)', '(\"starts\", object)')
t = t.replace('(\"stimuli\", np.object)', '(\"stimuli\", object)')
t = t.replace('.astype(np.int)', '.astype(np.int64)')
open(f, 'w', encoding='utf-8').write(t)
print('patched moabb neiry.py')
"

echo "Setup complete. Run: python -m pytest"
