docker build -t solar-model-build .
docker run -dp 7070:7070 --name solar-api solar-model-build