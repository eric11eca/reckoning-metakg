NUM=1

CURRENT=${NUM}
IMAGE_NAME=meta_kg
DOCKERFILE_NAME=Dockerfile

#GIT_HASH=`git log --format="%h" -n 1`
#IMAGE=$IMAGE_NAME_$USER-$GIT_HASH
IM_NAME=${IMAGE}_${NUM}

echo "Building $IMAGE"
docker buildx build --platform linux/amd64 --load -f $DOCKERFILE_NAME -t ic-registry.epfl.ch/nlp/$IMAGE --cache-from type=local,src=../../.docker_cache --cache-to type=local,mode=max,dest=../../.docker_cache .

echo ${IMAGE}