# Infogan Pytorch

An implementation of Infogan [1] for pytorch. For an explanation of the project see [2].

### Training

You can train a new model using `python trainer.py`.

### Pretrained Models

Some pretrained models are included with the code for inference. You can pass them to the network using the `--modelpath` argument in test.py.

[MNIST](https://drive.google.com/drive/folders/10P6CB9G8EdaJRo6J8-CgMFzfZmznoARJ?usp=sharing)  
[FashionMNIST](https://drive.google.com/drive/folders/1Zf1Pda0dLy4c8ZVJ470jXzla8iEmc58_?usp=sharing)

### Testing

You can test a newly trained model or one of the pretrained moedls with `python test.py --expname EXPNAME --modelpath "PATH_TO_PRETRAINED_MODELS" --modelname "NAME_OF_GENERATOR"`.

### References

[1] - [X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, P. Abbeel - InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)  
[2] - [Deepankar C. - Generation of Images via Attribute Manipulation using Disentangled Representation Learning](https://deepankarc.github.io//2020/02/04/infogan/)
