import tensorflow as tf
import numpy as np
import neural_structured_learning as nsl
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Define parameters
input_shape = [28, 28, 1]
num_classes = 10
conv_filters = [32, 64, 64]
kernel_size = (3, 3)
pool_size = (2, 2)
num_fc_units = [64]
batch_size = 32
epochs = 5
adv_multiplier = 0.1
adv_step_size = 0.2
adv_grad_norm = 'infinity'

# Load the MNIST dataset
data_train, data_test = tfds.load('mnist', split=['train', 'test'])

# Normalize the data
def normalize(features):
    features['image'] = tf.cast(features['image'], dtype=tf.float32) / 255.0
    return features

def convert_to_tuples(features):
    return features['image'], features['label']

def convert_to_dictionaries(image, label):
    return {'image': image, 'label': label}

data_train = data_train.map(normalize).shuffle(10000).batch(batch_size).map(convert_to_tuples)
data_test = data_test.map(normalize).batch(batch_size).map(convert_to_tuples)

# Build the base model
def build_model():
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name='image')
    x = inputs
    for i, num_filters in enumerate(conv_filters):
        x = tf.keras.layers.Conv2D(num_filters, kernel_size, activation='relu')(x)
        if i < len(conv_filters) - 1:
            x = tf.keras.layers.MaxPooling2D(pool_size)(x)
    x = tf.keras.layers.Flatten()(x)
    for num_units in num_fc_units:
        x = tf.keras.layers.Dense(num_units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model_base = build_model()
model_base.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model_base.summary()

# Train the base model

model_base_history = model_base.fit(data_train, epochs=epochs)
results = model_base.evaluate(data_test)
named_results = dict(zip(model_base.metrics_names, results))
print('\naccuracy:', named_results['sparse_categorical_accuracy'])

# Build the adversarial model
adv_config = nsl.configs.make_adv_reg_config(multiplier=adv_multiplier, 
        adv_step_size=adv_step_size, 
        adv_grad_norm=adv_grad_norm)

base_adv_model = build_model()
model_adv = nsl.keras.AdversarialRegularization(
    base_adv_model,
    label_keys=['label'],
    adv_config=adv_config)


data_train_adv = data_train.map(convert_to_dictionaries)
data_test_adv = data_test.map(convert_to_dictionaries)

# Train the adversarial model
model_adv.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model_adv_history = model_adv.fit(data_train_adv, epochs=epochs)
adv_results = model_adv.evaluate(data_test_adv)
named_adv_results = dict(zip(model_adv.metrics_names, adv_results))
print('\naccuracy:', named_adv_results['sparse_categorical_accuracy'])

# Plot the results
plt.plot(model_base_history.history['sparse_categorical_accuracy'])
plt.plot(model_adv_history.history['sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['base', 'adversarial'], loc='upper left')
plt.show()

ref_model = nsl.keras.AdversarialRegularization(
    model_base,
    label_keys=['label'],
    adv_config=adv_config)

ref_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

models_to_evaluate = {
    'base': model_base,
    'adversarial': model_adv.base_model,
}

metrics = {
    name: tf.keras.metrics.SparseCategoricalAccuracy()
    for name in models_to_evaluate.keys()
}

perturbed_imgs,labels,predictions = [],[],[]

for batch in data_test_adv:
    perturbed_batch = ref_model.perturb_on_batch(batch)
    perturbed_batch['image'] = tf.clip_by_value(perturbed_batch['image'], 0, 1)

    y_true = perturbed_batch.pop('label')
    perturbed_imgs.append(perturbed_batch['image'].numpy())
    labels.append(y_true.numpy())
    predictions.append({})

    for name, model in models_to_evaluate.items():
        y_pred = model(perturbed_batch)
        metrics[name].update_state(y_true, y_pred)
        predictions[-1][name] = tf.argmax(y_pred, axis=-1).numpy()

for name, metric in metrics.items():
    print(f'{name} accuracy: {metric.result().numpy()}')

batch_index = 5

batch_images = perturbed_imgs[batch_index]
batch_labels = labels[batch_index]
batch_predictions = predictions[batch_index]

n_columns = 5
n_rows = (batch_size + n_columns - 1) // n_columns

print('acc in batch %d: ' % batch_index, end='')
for name,pred in batch_predictions.items():
    print('%s model: %d / %d' % (name, np.sum(batch_labels == pred), batch_size))

plt.figure(figsize=(n_columns * 2, n_rows * 2))
for i,(img,y) in enumerate(zip(batch_images, batch_labels)):
    y_base = batch_predictions['base'][i]
    y_adv = batch_predictions['adversarial'][i]
    plt.subplot(n_rows, n_columns, i + 1)
    plt.title('true: %d, base: %d, adv: %d' % (y, y_base, y_adv))
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save the model

model_adv.save('Model/advs_model.h5')
model_base.save('Model/base_model.h5')
