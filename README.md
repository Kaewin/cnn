[keras documentation - models](https://keras.io/api/applications/)

1. **Imports:**
   ```python
   import tensorflow as tf
   import numpy as np
   from tensorflow import keras
   print(tf.__version__)
   ```

2. **Define and Compile the Neural Network:**
   ```python
   model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
   model.compile(optimizer='sgd', loss='mean_squared_error')
   ```

3. **Providing the Data:**
   ```python
   xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
   ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
   ```

4. **Training the Neural Network:**
   ```python
   model.fit(xs, ys, epochs=500)
   ```

5. **Making Predictions:**
   ```python
   print(model.predict([10.0]))
   ```

**Computer Vision Lab:**

This code explores a computer vision example using the Fashion MNIST dataset, focusing on code components:

1. **Importing TensorFlow:**
   ```python
   import tensorflow as tf
   print(tf.__version__)
   ```

2. **Loading and Exploring the Dataset:**
   ```python
   (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
   ```

3. **Data Preprocessing:**
   ```python
   training_images = training_images / 255.0
   test_images = test_images / 255.0
   ```

4. **Building the Neural Network Model:**
   ```python
   model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                       tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                       tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
   ```

5. **Compiling and Training the Model:**
   ```python
   model.compile(optimizer=tf.optimizers.Adam(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   model.fit(training_images, training_labels, epochs=5)
   ```

6. **Model Evaluation:**
   ```python
   model.evaluate(test_images, test_labels)
   ```

**Callbacks API Implementation:**

This ungraded lab demonstrates the Callbacks API in TensorFlow, focusing on key code components:

1. **Load and Normalize the Fashion MNIST Dataset:**
   ```python
   import tensorflow as tf
   fmnist = tf.keras.datasets.fashion_mnist
   (x_train, y_train), (x_test, y_test) = fmnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0
   ```

2. **Creating a Callback Class:**
   ```python
   class myCallback(tf.keras.callbacks.Callback):
       def on_epoch_end(self, epoch, logs={}):
           if(logs.get('loss') < 0.4):
               print("\nLoss is lower than 0.4 so cancelling training!")
               self.model.stop_training = True
   callbacks = myCallback()
   ```

3. **Define and Compile the Model:**
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
   ```python
   model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
   ```

5. **Optional Challenge:**
   ```python
   if(logs.get('accuracy') > 0.6):
       print("\nAccuracy exceeds 60%, stopping training!")
       self.model.stop_training = True
   ```

**Improving Computer Vision Accuracy:**

This lab explores the improvement of computer vision accuracy using shallow and convolutional neural networks:

**Shallow Neural Network:**
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

**Exploration of Convolutions:**

This lab explores convolutions by creating a basic convolution on a 2D grayscale image:

**Image Loading and Visualization:**
   ```python
   from scipy.datasets import ascent
   ascent_image = ascent()
   import matplotlib.pyplot as plt
   plt.grid(False)
   plt.gray()
   plt.axis('off')
   plt.imshow(ascent_image)
   plt.show()
   ```

**Convolution and Max Pooling:**
   ```python
   filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
   weight = 1
   # Convolution code...
   # Visualization code...
   # Max pooling code...
   ```