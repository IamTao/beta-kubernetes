SCALE=$(s)

setup:
	# delete existing worker.
	$(MAKE) delete_cluster
	# create cluster.
	kubectl create -f dmlb-master.yaml
	kubectl create -f dmlb-controller.yaml

scale:
	kubectl scale rc dmlb-worker --replicas=${SCALE}

configure:
	# configre cluster.
	python init.py
	kubectl cp hostfile dmlb-master:/mlodata1/tlin/

login:
	kubectl exec -it dmlb-master -- bash

delete_cluster:
	-kubectl delete -f dmlb-master.yaml
	-kubectl delete -f dmlb-controller.yaml

get_pods:
	kubectl get pods | grep dmlb
