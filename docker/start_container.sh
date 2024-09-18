docker build -t jp/senpamae \
  --build-arg USER_ID=$(id -u) \
  --build-arg HOST_UID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .

docker run -it --gpus all --name senpamae --ipc=host \
-v /home/jprexl/Code/SenPa-MAE/src:/home/user/src/ \
-v /home/jprexl/Data/:/home/user/data/ \
-v /home/jprexl/Results/:/home/user/results/ \
jp/senpamae
