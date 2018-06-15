#!/bin/bash

SOURCE=$1
DEST=$2
SAMPLE_FILE=$3
if [ "${4}" = "build-dir" ]; then
echo "Getting unique directories ${ALPS_APP_PE}"
LOCAL_DIRS=$(python -c "import numpy as np
import re
with open('${SAMPLE_FILE}') as f:
    dirs = [re.split(' |_', l.strip())[1] for l in f]

w_size = ${PBS_NUM_NODES}*${PBS_NUM_PPN}
rank = ${ALPS_APP_PE}
local_dirs = np.array_split(np.array(dirs), w_size)[rank]
print(' '.join(local_dirs))
")
echo "Creating unique directories ${ALPS_APP_PE}"
for i in ${LOCAL_DIRS}
do 
CLASS_DIR=${DEST}/${i}
mkdir -p $CLASS_DIR
done
else
echo "Spliting up files ${ALPS_APP_PE}"
LOCAL_FILES=$(python -c "import numpy as np
import re
with open('${SAMPLE_FILE}') as f:
    dir_file_list = [re.split(' |_', l.strip())[1:] for l in f]
dir_file_list = [''.join([l[0],'-', l[0],'_', l[1], '.JPEG'])  for l in dir_file_list]
w_size = ${PBS_NUM_NODES}*${PBS_NUM_PPN}
rank = ${ALPS_APP_PE}
local_files = np.array_split(np.array(dir_file_list), w_size)[rank]
print(' '.join(local_files))
")
echo "moving files ${ALPS_APP_PE}"
for i in ${LOCAL_FILES}
do 
DIR=$(echo ${i} | cut -d '-' -f 1)
FILE=$(echo ${i} | cut -d '-' -f 2)
cp ${SOURCE}/${FILE} ${DEST}/${DIR}/${FILE}
done
fi
