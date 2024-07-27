from pathlib import Path
import os

#######################################################
######## getting the directory of the file ############
#######################################################

home = Path().resolve().as_posix()
data = os.listdir(home)[5]
data_dir = home + '/' + data


