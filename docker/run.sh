. ./env.sh

read -p "Enter port number: " PORT

echo ""
echo "Starting container $CONTAINER_NAME from image $IMAGE_NAME ..."
echo ""

if [ -z ${DATA_DIR} ]; then
    docker run -d \
        --shm-size=1g --privileged=true \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -p ${PORT}:8000 \
        -v ${PROJECT_DIR}:/project/ \
        --name ${CONTAINER_NAME} ${IMAGE_NAME}:latest
else
    docker run -d \
        --shm-size=1g --privileged=true \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -p ${PORT}:8000 \
        -v ${PROJECT_DIR}:/project/ \
        -v ${DATA_DIR}:/data/
        --name ${CONTAINER_NAME} ${IMAGE_NAME}:latest
fi

docker ps | grep ${CONTAINER_NAME} 

