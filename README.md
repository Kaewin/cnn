text

[keras documentation - models](https://keras.io/api/applications/)


**Summary 1:**

This code provides a hands-on introduction to deep learning using a simple example of training a neural network to learn the relationship between two numbers. Emphasizing the key code components, the process involves:

1. **Imports:**
   - TensorFlow, NumPy, and Keras are imported to facilitate neural network creation.
   ```python
   import tensorflow as tf
   import numpy as np
   from tensorflow import keras
   print(tf.__version__)
   ```

2. **Define and Compile the Neural Network:**
   - A basic neural network with one layer and one neuron is created using the Sequential class in Keras.
   ```python
   model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
   ```
   - Compilation involves specifying the loss function (mean squared error) and the optimizer (stochastic gradient descent).
   ```python
   model.compile(optimizer='sgd', loss='mean_squared_error')
   ```

3. **Providing the Data:**
   - Input-output pairs (X and Y values) are defined using NumPy arrays.
   ```python
   xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
   ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
   ```

4. **Training the Neural Network:**
   - The model is trained using the `model.fit()` method, iterating through epochs to minimize the loss and improve predictions.
   ```python
   model.fit(xs, ys, epochs=500)
   ```

5. **Making Predictions:**
   - The trained model is used to predict the output for a new input (e.g., x=10).
   ```python
   print(model.predict([10.0]))
   ```
   - The result illustrates the probabilistic nature of neural networks, where predictions are based on learned relationships rather than certainties.

**Summary 2:**

This code presents an ungraded lab that delves into a computer vision example beyond the traditional "Hello World" scenario. The emphasis is on understanding the code components involved in building a neural network for classifying different items of clothing using the Fashion MNIST dataset.

1. **Importing TensorFlow:**
   - TensorFlow is imported for building the neural network.
   ```python
   import tensorflow as tf
   print(tf.__version__)
   ```

2. **Loading and Exploring the Dataset:**
   - The Fashion MNIST dataset, consisting of grayscale 28x28 pixel clothing images with associated labels, is loaded using `tf.keras.datasets.fashion_mnist`.
   ```python
   (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
   ```
   - The labels correspond to different clothing types.

3. **Data Preprocessing:**
   - Values in the image arrays are normalized to a range between 0 and 1 to enhance neural network training.
   ```python
   training_images = training_images / 255.0
   test_images = test_images / 255.0
   ```

4. **Building the Neural Network Model:**
   - A classification model is defined using Keras with layers like Flatten, Dense, and activation functions such as ReLU and Softmax.
   ```python
   model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                       tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                       tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
   ```

5. **Compiling and Training the Model:**
   - The model is compiled with an optimizer, loss function, and metrics, then trained using the training data.
   ```python
   model.compile(optimizer=tf.optimizers.Adam(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   model.fit(training_images, training_labels, epochs=5)
   ```

6. **Model Evaluation:**
   - The accuracy of the trained model is evaluated on the test dataset using `model.evaluate()`.
   ```python
   model.evaluate(test_images, test_labels)
   ```
   - The achieved accuracy provides insights into the model's performance on unseen data.

The code walkthrough helps understand the process of building, training, and evaluating a neural network for a computer vision task, with an emphasis on image classification.

**Summary: 3**

This ungraded lab explores the implementation of the Callbacks API in TensorFlow to control training by halting it when a specified metric, such as loss, is met. The key code components are emphasized in the walkthrough:

1. **Load and Normalize the Fashion MNIST Dataset:**
   - TensorFlow is used to load the Fashion MNIST dataset, and pixel values are normalized to optimize training.
   ```python
   import tensorflow as tf
   fmnist = tf.keras.datasets.fashion_mnist
   (x_train, y_train), (x_test, y_test) = fmnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0
   ```

2. **Creating a Callback Class:**
   - A custom callback class is created by inheriting from `tf.keras.callbacks.Callback`. The `on_epoch_end()` method is defined to check the loss at the end of each training epoch and stop training if the loss falls below 0.4.
   ```python
   class myCallback(tf.keras.callbacks.Callback):
       def on_epoch_end(self, epoch, logs={}):
           if(logs.get('loss') < 0.4):
               print("\nLoss is lower than 0.4 so cancelling training!")
               self.model.stop_training = True
   callbacks = myCallback()
   ```

3. **Define and Compile the Model:**
   - A model is defined with a structure similar to the previous lab. The architecture includes a Flatten layer, a Dense layer with ReLU activation, and a Dense layer with softmax activation. The model is compiled with specified optimizer, loss, and metrics.
   ```python
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
   ])
   model.compile(optimizer=tf.optimizers.Adam(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   ```

4. **Train the Model with Callback:**
   - The model is trained with the defined callback, and the training stops if the loss falls below 0.4 during any epoch.
   ```python
   model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
   ```

5. **Optional Challenge:**
   - A challenge is provided to modify the code to stop training when the accuracy metric exceeds 60%. This encourages experimentation with the callback for different conditions.
   ```python
   # Optional Challenge
   # Stop training when accuracy exceeds 60%
   if(logs.get('accuracy') > 0.6):
       print("\nAccuracy exceeds 60%, stopping training!")
       self.model.stop_training = True
   ```

This exercise showcases the flexibility of callbacks in TensorFlow, allowing dynamic control over the training process based on specified conditions.

**Summary 4:**

This ungraded lab explores the improvement of computer vision accuracy using convolutional neural networks (CNNs) compared to a shallow neural network. The key code components and the impact on accuracy are highlighted:

**Shallow Neural Network:**

1. **Load and Normalize the Fashion MNIST Dataset:**
   - Fashion MNIST dataset is loaded, and pixel values are normalized.
   ```python
   import tensorflow as tf
   fmnist = tf.keras.datasets.fashion_mnist
   (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
   training_images = training_images / 255.0
   test_images = test_images / 255.0
   ```

2. **Define and Train the Shallow Neural Network:**
   - A shallow neural network with a Flatten layer, a Dense layer with ReLU activation, and a Dense layer with softmax activation is defined and trained.
   ```python
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation=tf.nn.relu),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
   ])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(training_images, training_labels, epochs=5)
   test_loss = model.evaluate(test_images, test_labels)
   ```

**Convolutional Neural Network:**

3. **Implementing Convolutional Layers:**
   - Convolutional layers and max-pooling layers are added before the dense layers to focus and highlight specific features in the images.
   ```python
   model = tf.keras.models.Sequential([
     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(10, activation='softmax')
   ])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(training_images, training_labels, epochs=5)
   test_loss = model.evaluate(test_images, test_labels)
   ```

4. **Visualizing Convolutions and Pooling:**
   - Convolutional output is visualized for specific images to observe common features highlighted by the convolution/pooling combination.
   ```python
   # Visualization code provided in the lab
   ```

**Exercises:**

- **Edit Convolutions:**
  - Change the number of convolutions (e.g., from 32 to 16 or 64) and observe the impact on accuracy and training time.

- **Remove Final Convolution:**
  - Remove the final convolution and observe the impact on accuracy and training time.

- **Add More Convolutions:**
  - Experiment with adding more convolutional layers and observe the impact.

- **Remove Convolutions but First:**
  - Remove all convolutions except the first and observe the impact.

- **Implement Callback:**
  - Implement a callback to check the loss function and cancel training once it reaches a certain threshold.

This exercise encourages experimentation with CNN architecture, providing insights into how different configurations affect accuracy and training performance.

**Summary 5:**

In this ungraded lab, the exploration of convolutions is demonstrated by creating a basic convolution on a 2D grayscale image using SciPy's ascent image. The key code components and the effects of convolution and max pooling are highlighted:

**Image Loading and Visualization:**

1. **Load the Ascent Image:**
   - The ascent image from SciPy is loaded.
   ```python
   from scipy.datasets import ascent
   ascent_image = ascent()
   ```

2. **Visualize the Image:**
   - The pyplot library is used to visualize the ascent image.
   ```python
   import matplotlib.pyplot as plt
   plt.grid(False)
   plt.gray()
   plt.axis('off')
   plt.imshow(ascent_image)
   plt.show()
   ```

**Convolution:**

3. **Create a Filter:**
   - A 3x3 filter is created and experimented with different values.
   ```python
   filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
   weight = 1
   ```

4. **Apply Convolution:**
   - Convolution is applied by iterating over the image, multiplying pixel values by filter values, applying weight, and ensuring the result is within the range of 0-255.
   ```python
   for x in range(1, size_x-1):
     for y in range(1, size_y-1):
       # Convolution calculation
       # ...
       # Load into the transformed image
       image_transformed[x, y] = convolution
   ```

5. **Plot Transformed Image:**
   - The transformed image after convolution is plotted.
   ```python
   plt.gray()
   plt.grid(False)
   plt.imshow(image_transformed)
   plt.show()
   ```

**Effect of Max Pooling:**

6. **Apply Max Pooling:**
   - Max pooling with a (2, 2) pool is applied to reduce the image dimensions, preserving features by selecting the maximum pixel value in each pool.
   ```python
   # Iteration for max pooling
   # ...
   # Plot the reduced image
   plt.gray()
   plt.grid(False)
   plt.imshow(newImage)
   plt.show()
   ```

**Exercises:**

- **Experiment with Filters:**
  - Try different filter values and observe their effects on the transformed image.

- **Adjust Convolution Weight:**
  - Experiment with different weights for the convolution and observe the impact.

- **Explore Pooling:**
  - Try different pooling sizes (e.g., (3, 3)) and observe the impact on the reduced image.

This lab provides a hands-on exploration of convolutional operations and their effects on image transformation and dimensionality reduction.