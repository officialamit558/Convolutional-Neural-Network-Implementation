# Convolutional-Neural-Network-Implementation

# Forward Propagation
In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the Functional API, you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

input_img = tf.keras.Input(shape=input_shape):
Then, create a new node in the graph of layers by calling a layer on the input_img object:

tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img): Read the full documentation on Conv2D.

tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'): MaxPool2D() downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window. For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on MaxPool2D.

tf.keras.layers.ReLU(): computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on ReLU.

tf.keras.layers.Flatten(): given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.

If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where  𝑘=ℎ×𝑤×𝑐
 . "k" equals the product of all the dimension sizes other than the first dimension.

tf.keras.layers.Dense(units= ... , activation='softmax')(F): given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on Dense.

In the last function above (tf.keras.layers.Dense()), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer):

outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)
Window, kernel, filter, pool
The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels.

This is why the parameter pool_size refers to kernel_size, and you use (f,f) to refer to the filter size.

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place.

