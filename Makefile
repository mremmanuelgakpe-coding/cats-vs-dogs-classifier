.PHONY: install data weights features train evaluate test api demo docker-build docker-run k8s-validate k8s-deploy

PY ?= .venv/bin/python
export PYTHONPATH := src

install:            ## environnement d'entraînement (Python 3.12+)
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:               ## miroir GitHub du jeu Kaggle, épinglé à un commit, puis contrôle qualité et découpage
	git clone https://github.com/laxmimerit/dog-cat-full-dataset.git data/raw/src_repo || true
	git -C data/raw/src_repo checkout b2d738611a3daada177ea755c11b4be14726d38c
	$(PY) -m catdog.data

weights:            ## poids ImageNet officiels (GitHub), vérifiés par empreinte SHA-256 au chargement
	mkdir -p weights
	cd weights && curl -fsSLO https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 \
	 && curl -fsSLO https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
	 && curl -fsSLO https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
	 && curl -fsSLO https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5 \
	 && curl -fsSLO https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5

features:           ## extraction des caractéristiques (VGG16, ResNet50, InceptionV3, MobileNetV2)
	$(PY) -m catdog.features

train: features     ## têtes (grid + random search), fine-tuning MobileNetV2, CNN maison
	$(PY) -m catdog.heads
	$(PY) -m catdog.nets finetune
	$(PY) -m catdog.nets cnn

evaluate:           ## test final, choix du modèle, export ONNX, figures
	$(PY) -m catdog.evaluate
	$(PY) -m catdog.plots

test:
	$(PY) -m pytest -q

api:
	set -a && . ./.env && set +a && $(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000

demo:
	$(PY) -m streamlit run demo/app.py --server.address 127.0.0.1

docker-build:
	docker build -t cats-vs-dogs-api:1.0.0 .

docker-run:
	docker run --rm -p 127.0.0.1:8000:8000 --env-file .env --read-only --tmpfs /tmp --cap-drop ALL \
	  --security-opt no-new-privileges cats-vs-dogs-api:1.0.0

k8s-validate:
	for f in k8s/*.yaml; do case $$f in *kustomization*) ;; *) $(PY) -m kubernetes_validate --strict -k 1.31.0 $$f ;; esac; done

k8s-deploy:         ## cluster local kind : image chargée sans registre, clé créée comme secret
	kind load docker-image cats-vs-dogs-api:1.0.0
	kubectl apply -f k8s/namespace.yaml
	kubectl -n catdog create secret generic catdog-api --from-literal=api-key="$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -k k8s/
	kubectl -n catdog set image deployment/catdog-api api=cats-vs-dogs-api:1.0.0
