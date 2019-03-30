. ./env.sh

echo ""
echo "Removing image $IMAGE_NAME:latest ..."
echo ""

docker image rm -f $IMAGE_NAME:latest
