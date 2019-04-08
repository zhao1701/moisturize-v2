. ./env.sh

while [ -z $GPU ]; do
    read -p "Enter GPU number: " GPU
done

read -p "Enter port number: " PORT

echo ""
echo "Starting container $CONTAINER_NAME on GPU $GPU from image $IMAGE_NAME ..."
echo ""

if [ -z ${DATA_DIR} ]; then
    nvidia-docker run -d \
        --privileged=true \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e CUDA_VISIBLE_DEVICES=${GPU} \
        -p ${PORT}:8000 \
        -v ${PROJECT_DIR}:/project/ \
        --name ${CONTAINER_NAME} ${IMAGE_NAME}:latest
else
    nvidia-docker run -d \
        --privileged=true \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e CUDA_VISIBLE_DEVICES=${GPU} \
        -p ${PORT}:8000 \
        -v ${PROJECT_DIR}:/project/ \
        -v ${DATA_DIR}:/data/
        --name ${CONTAINER_NAME} ${IMAGE_NAME}:latest
fi

docker ps | grep ${CONTAINER_NAME} 

