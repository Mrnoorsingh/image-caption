# Image Captioning using CNN-LSTM Architecture

## Overview
The task of this project is to take an image as input and describe it in human readable text. The model is a type of One-to-Many mode of RNN, that is, it 
translates a single image to a sequence of words. Flickr8k Dataset has been used to train the model.

## Architecture
The model can be divided into 3 modules
* **Image Encoder**
  It takes an image as input and encodes it into a rich fixed-size vector representation. In this project, Resnet50 CNN architecture pretrained on ImageNet
  dataset is utilized to encode the images.

* **Language Encoder**
  Language encoder or text encoder encodes the input text into a fixed-size vector using an Embedding and a LSTM layer.
  An Embedding layer is a weight matrix which encodes each word(in hot vector form) in the vocabulary into a unique fixed-length dense vector representation. 
  
* **Decoder**
  The decoder takes the extracted feature vector of image and encoded partial text sequence and predicts the next word of for the sequence.
  The LSTM decoder outputs the probability distribution of next word. Further, this Probability distribution is used by *Beam Search* decoder to construct a human readable sentence
  
 
 
 
 
<p align="center">
  <img width="460" height="600" src="https://github.com/Mrnoorsingh/image-caption/blob/master/images/model%20.png", title="Architecture" />
</p>

### Output Examples

<p>
    <img src="/images/boat.png" alt>
</p>
<p>
    <em>a person kayaking in a body of water .</em>
</p>


<p>
    <img src="/images/dog.png" alt>
</p>
<p>
    <em>a small white dog catching a frisbee in its mouth .</em>
</p>


<p>
    <img src="/images/soccer.png" alt>
</p>
<p>
    <em>a small boy in a red grass field is after a soccer ball .</em>
</p>


<p>
    <img src="/images/trek.png" alt>
</p>
<p>
    <em>a man stands on a mountaintop .</em>
</p>

<p>
    <img src="/images/rock.png" alt>
</p>
<p>
    <em>a man wearing a harness climbs a rock face .</em>
</p>

## Evaluation
The model has achieved **BLEU** Score of almost ```0.49```. Bleu Score is a metric used to compare generated sentences to their corresponding reference sentences. A perfect match results in bleu score of 1.0 whereas a perfect mismatch results in bleuscore of 0.0

## References
https://arxiv.org/abs/1411.4555

https://cs.stanford.edu/people/karpathy/sfmltalk.pdf

http://karpathy.github.io/2015/05/21/rnn-effectiveness/




