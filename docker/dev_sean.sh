#!/bin/sh
docker rm -f $(docker ps -a --filter "name=storyviewer-code" -q)

# 创建代码镜像，代码镜像禁止使用latest,防止不小心推送到线上
docker build --rm -f docker/DevDockerfile.CodeUS -t registry-vpc.us-west-1.aliyuncs.com/zingfront/storyviewer-code:test . --no-cache

# 推送代码镜像
# 测试模式不推送代码镜像
# docker push registry-vpc.us-west-1.aliyuncs.com/zingfront/storyviewer-code:latest

# 服务器运行（首先在服务器上，将代码镜像pull下来）
docker run --name storyviewer-code-`date "+%Y-%m-%d-%H-%M"` -d -p 10132:80 --restart=always -v /nas/zbase/:/nas/zbase/ -v /home/dev/webspace/storyviewer_api/app:/code/app registry-vpc.us-west-1.aliyuncs.com/zingfront/storyviewer-code:test