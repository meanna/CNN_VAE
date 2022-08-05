# CNN-VAE

## install
- pytorch
- python 3.7
- CLIP

## load dataset
- load CelebA

## train

- load CLIP embedding (embeddings.csv)
- go to train.py
  - set path in CelebA_CLIP
  - go to get_data_celebA_small and get_data_celebA, set dataset paths
  - go to main function and adjust
  - choose VAE model you want to use, default is line 99
        - you can use "RES_VAE_conditioned.py" by importing it there
        - e.g. from RES_VAE_conditioned import VAE
  - set the parameters
    - batch_size = 128 (suggested is 256)
    - num_epoch = 10
    - dataset = "celeba_small"
    - latent_dim = 128 (suggested is 512)
    - load_checkpoint = None or use checkpoint name
    - run_train = True

## generate

(works only when using RES_VAE_condition....)

- go to train.py, in main function, you can run
  - image_generation_clip(target_attr="funny", save_path=result_folder) : generate with CLIP text embedding as condition
  - image_generation_with_condition(test_labels) : the sampled input tensor has size input + condition
  - image_generation() : generate from sampled input tensor with image embedding condition
  - image_generation_zero() : the condition is zero tensor