from pathlib import Path
import os

#######################################################
######## getting the directory of the file ############
#######################################################

home = Path().resolve().parent.as_posix() + '/data'
data = os.listdir(home)[0]
data_dir = home + '/' + data

