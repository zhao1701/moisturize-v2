. ./env.sh

echo ""
if [ -z ${REGISTRY} ]; then
    REGISTRY="public registry"
fi
echo "Pushing image $IMAGE_NAME to ${REGISTRY} ..."

docker push $IMAGE_NAME
