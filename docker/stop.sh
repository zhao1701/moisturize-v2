. ./env.sh

echo ""
echo "Stopping and removing container $CONTAINER_NAME ..."
echo ""

docker container rm -f $CONTAINER_NAME
