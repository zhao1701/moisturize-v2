#!/bin/bash

FILE_ID=0B7EVK8r0v71pZjFTYXZWM3FlRnM
DOWNLOAD_DESTINATION='celeb-a.zip'
UNZIP_DESTINATION='train'

echo 'Downloading data...'
if [ ! -f ${DOWNLOAD_DESTINATION} ]; then
    python download-google-drive.py ${FILE_ID} ${DOWNLOAD_DESTINATION}
fi

if [ ! -d ${UNZIP_DESTINATION} ]; then
    unzip ${DOWNLOAD_DESTINATION}
    mv img_align_celeba ${UNZIP_DESTINATION}
else
    echo "Destination folder for Celeb-A dataset already exists."
fi

echo 'Cleaning up...'
rm ${DOWNLOAD_DESTINATION}
