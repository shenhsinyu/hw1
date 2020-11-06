## hw1 
####  car brand classification


### Hardware
- Ubuntu 18.04
- GeForce GTX 1080


### Environment
- Pytorch


### Prepare images
- After downloading the data then run get_data.py to convert data to this structure:

            |+-  hw1
            |    +- train
            |          +-class_1
            |          +-class_2
                            …
            |          +-class_196

            |+-  hw1
            |    +- test
            |          +-000004.jpg
            |          +-000005.jpg
                            …

### Training and testing
- Run train.py and change the path you want to save the model and prediction.
