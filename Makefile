HOST=lin@iccluster029.iccluster.epfl.ch
DISK=~/


ssh_server:
	ssh ${HOST}

cp_gpu:
	rsync -av -e ssh ../beta-kubernetes ${HOST}:${DISK}

build_image:
	cd images && docker-compose build


push_image:
	cd images && bash push_image.sh
