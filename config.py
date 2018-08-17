num_classes = 103
num_epochs = 200
batch_size = 256
patience = 36
learning_rate = 0.0008
decay = 0.9
feature_shape = [None, 7, 7, 2048]
image_size = 224
inception_resnet_model = './model_inc/inc_res.ckpt'
resnet_model = './model/resnet.ckpt'
new_data_resnet = 'new_data.pickle'
new_data_inception = 'new_data_inc.pickle'
