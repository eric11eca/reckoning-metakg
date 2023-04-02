NUM=step2_5

CURRENT=${NUM}
IMAGE_NAME=meta_kg
DOCKERFILE_NAME=docker/update_train/Dockerfile

GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH
IM_NAME=${IMAGE_NAME}_${NUM}

echo "Building $IM_NAME"
# docker buildx build --build-arg DUMMY='cloud' --platform linux/amd64 --load -f $DOCKERFILE_NAME -t $IM_NAME --cache-from type=local,src=../../.docker_cache --cache-to type=local,mode=max,dest=../../.docker_cache .

docker build -f $DOCKERFILE_NAME . -t $IM_NAME --build-arg DUMMY='cloud' --platform linux/amd64

echo "Pushing $IM_NAME to Harbor"
docker tag $IM_NAME eric11eca/$IM_NAME
docker push eric11eca/$IM_NAME

# runai submit --name meta-kg-clutrr -i eric11eca/meta_kg_8 --interactive --attach -g 1 --node-type G10
# runai submit --name meta-kg-proof5 -i eric11eca/meta_kg_8 --attach -g 1 --node-type G10
# runai submit --name meta-kg-clutrr6 -i eric11eca/meta_kg_9 --attach -g 1 --node-type G10
# runai submit --name meta-kg-proof2-d4 -i eric11eca/meta_kg_2 --attach -g 1 --node-type G10