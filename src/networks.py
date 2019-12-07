from tensorflow import keras
from tensorflow.keras.layers import concatenate, Conv2D, Dense, Flatten, Layer, MaxPooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3

def network_factory(class_type, filters, dim_output, dim_input=None):
    if class_type == "Inception": return InceptionNet(filters, dim_output)
    if class_type == "BaseConv": return BaseConvNet(filters, dim_output)
    if class_type == "InceptionV3": return InceptionV3(weights=None, classes=dim_output)

    raise Exception("Class type was not recognized.")

class ConvNet(keras.Model):
    def __init__(self):
       super(keras.Model, self).__init__()

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x


class InceptionLayer(Layer):
    def __init__(self, filters, activation):
        super(InceptionLayer, self).__init__()

        self.conv_1_1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=activation)
        self.conv_2_1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=activation)
        self.conv_2_2 = Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation=activation)
        self.conv_3_1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=activation)
        self.conv_3_2 = Conv2D(
            filters=filters,
            kernel_size=5,
            padding="same",
            activation=activation)
        self.maxpool_4_1 = MaxPooling2D(
            pool_size=3,
            strides=1,
            padding="same")
        self.conv_4_2 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=activation)


    def call(self, x):
        x1 = self.conv_1_1(x)
        x2 = self.conv_2_2(self.conv_2_1(x))
        x3 = self.conv_3_2(self.conv_3_1(x))
        x4 = self.conv_4_2(self.maxpool_4_1(x))
        x = concatenate([x1, x2, x3, x4], axis=3)
        return x

class BaseConvNet(ConvNet):
    def __init__(self, filters, dim_output):
        super(ConvNet, self).__init__()
        self.model_layers = [
            Conv2D(
                filters=filters,
                activation="relu",
                kernel_size=3),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                filters=filters*2,
                activation="relu",
                kernel_size=3),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                filters=filters*3,
                kernel_size=3,
                activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                filters=filters*3,
                kernel_size=3,
                activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(
                units = 128,
                activation="relu"),
            Dense(
                dim_output,
                activation="softmax")
        ]

class InceptionNet(ConvNet):
    def __init__(self, filters, dim_output):
        super(ConvNet, self).__init__()
        self.model_layers = [
            InceptionLayer(
                filters=filters,
                activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            InceptionLayer(
                filters=filters,
                activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            InceptionLayer(
                filters=filters,
                activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            InceptionLayer(
                filters=filters,
                activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(
                units=128,
                activation="relu"),
            Dense(
                units=dim_output,
                activation="softmax")
        ]
