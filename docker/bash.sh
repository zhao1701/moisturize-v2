. ./env.sh

echo ""
echo "Starting ZSH session in container $CONTAINER_NAME ..."
echo ""

docker container exec -it $CONTAINER_NAME zsh
