mkdir -p ~/.docker/cli-plugins/

BUILDX_URL=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep browser_download_url | grep linux-amd64 | cut -d '"' -f 4)

curl -L $BUILDX_URL -o ~/.docker/cli-plugins/docker-buildx

chmod +x ~/.docker/cli-plugins/docker-buildx

DOCKER_BUILDKIT=1 docker build -t lmcache:latest -f ../docker/Dockerfile .