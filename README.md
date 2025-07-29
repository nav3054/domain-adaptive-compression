<!-- # domain-adaptive-compression -->
# Cross-Domain Neural Models for Learned Compression
A domain-adaptive learned image compression project spanning natural and satellite images using VAEs, residual models, and transformer-based hyperpriors.

This project explores domain-adaptive learned image compression across diverse visual domains, such as natural and satellite images. The goal is to build models that can retain high compression performance under domain shift.

## Completed Work
### Domain-Adaptive Residual VAE
- A residual VAE trained on MNIST and fine-tuned on EMNIST to handle low-to-moderate domain shift.
- Code : [Github repo - Domain-adaptive-Residual-VAE](https://github.com/nav3054/Domain-adaptive-Residual-VAE) 

## In Progress / Upcoming
### Domain adaptation on Ballé 2018 Compression Model
- A reimplementation of the "Variational Image Compression with a Scale Hyperprior" paper with plans to evaluate and adapt it across domains.
- Paper/Model implementation completed: [GitHub repo - balle2018-scale-hyperprior](https://github.com/nav3054/balle2018-scale-hyperprior)
- Domain adaptation experiments have not been conducted yet.

### Domain adaptation on Ballé 2018 + Transformer Hyperprior
- An extended version of the Ballé model using a Transformer-based hyperprior to explore attention-driven compression under domain shift.
- Model Implementation (In Progress) : [Github Repo - balle2018-transformer-hyperprior](https://github.com/nav3054/balle2018-transformer-hyperprior)
- Domain adaptation experiments are pending.
