#!/bin/sh

docker build --rm -f docker/Dockerfile.Base -t registry-vpc.us-west-1.aliyuncs.com/zingfront/fastapi-storyviewer-base:latest . --no-cache

# 检查镜像是否成功构建
if ! docker images -q registry-vpc.us-west-1.aliyuncs.com/zingfront/fastapi-storyviewer-base:latest > /dev/null; then
    echo "Error: fastapi-storyviewer-base:latest image failed to build."
    exit 1
fi

echo "Success: build fastapi-storyviewer-base:latest success."

# docker push registry-vpc.us-west-1.aliyuncs.com/zingfront/fastapi-kode-crypto-base