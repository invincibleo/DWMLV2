source activate py2SED2017
rm -f screenlog.0
KERAS_BACKEND=tensorflow screen -L python test_Dataset_AVEC2016_single.py
