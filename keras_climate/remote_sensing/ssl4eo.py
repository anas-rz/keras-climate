import keras
from keras_climate.remote_sensing.deeplabv3plus import resnet_backbone


def SSL4EOResNet50(input_shape=(224, 224, 13), name="ssl4eo_resnet50"):
    inputs = keras.Input(shape=input_shape, name="image")
    _, features = resnet_backbone(inputs, layer_counts=(3, 4, 6, 3), output_stride=32,
                                   name="backbone")
    return keras.Model(inputs, features, name=name)
