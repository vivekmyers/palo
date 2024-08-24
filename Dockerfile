FROM robonet-base:latest

RUN ~/myenv/bin/pip install tensorflow jax[cuda11_pip] flax distrax ml_collections h5py wandb tensorflow-text tensorflow-hub \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN ~/myenv/bin/pip install einops
RUN ~/myenv/bin/pip install transformers
ENV PYTHONPATH=${PYTHONPATH}:/home/robonet/code/jaxrl_minimal:/home/robonet/code/denoising-diffusion-flax

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

RUN sudo mkdir -p /usr/local/gcloud \
  && sudo tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && sudo /usr/local/gcloud/google-cloud-sdk/install.sh --quiet

ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN git config --global --add safe.directory /home/robonet/code/jaxrl_minimal

RUN echo "gcloud auth activate-service-account --key-file /tmp/gcloud_key.json" >> ~/.bashrc
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud_key.json

WORKDIR /home/robonet/code/jaxrl_minimal
