# Image Captioning with CNN-LSTM
This project aims to implement an image captioning application with the help of `DenseNet201` **CNN** architecture which assists in extracting features from input images(encoder) and passes them to an **Long Short Term Memory** decoder which outputs a sequence of tokens after having been trained on flickr8k dataset.

**<u>Project Link:</u>** https://huggingface.co/spaces/ViratChauhan/image-captioner-3-in-1


There exist many limitations for such a model as it will forever be prone to producing erroneous captions due to the inherent inability of LSTMs to interpret human language. Some solutions can be **Bahdanau, Luong Attention** layers which support the LSTM decoder in highlighting objects in the image as introduced in the Show, Attend and Tell paper published in 2015. Another solution is resorting to **Beam Search** which selects tokens based on top k predictions output by the model.

Have got a couple of changes lined up for this model which can help in creating this project more _fancier_, namely the implementation of a working attention layer which I failed at currently due to time constraints, usage of ViT transformers to extract comprehensible features and interpret them with decoders of transformer family such as GPT-2, and extend the usecase by adding translation assistance!