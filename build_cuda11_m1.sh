NUM=1

CURRENT=${NUM}
IMAGE_NAME=meta_kg
DOCKERFILE_NAME=Dockerfile

GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH
IM_NAME=${IMAGE_NAME}_${NUM}

echo "Building $IM_NAME"
docker buildx build --platform linux/amd64 --load -f $DOCKERFILE_NAME -t $IM_NAME --cache-from type=local,src=../../.docker_cache --cache-to type=local,mode=max,dest=../../.docker_cache .

echo "Pushing $IM_NAME to Harbor"
docker tag $IMAGE ic-registry.epfl.ch/nlp/$IMAGE
docker push ic-registry.epfl.ch/nlp/$IMAGE

# export KUBECONFIG=~/.kube/config_runai