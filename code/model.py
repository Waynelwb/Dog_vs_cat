import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# 定义模型：卷积
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# 配置模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

# 设置数据集路径
train_dir = '../data/new/train'
validation_dir = '../data/new/validation'

# 设置图片生成器，用于数据增强
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# 从指定路径获取数据，进行数据增强
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=128,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=128,
                                                        class_mode='binary')

# 设置早停策略，满足条件即终止训练
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# 训练模型
history = model.fit(train_generator,
                    steps_per_epoch=118,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=40,
                    callbacks=[early_stopping])

# 保存模型
model.save('../model/dogs_and_cats.h5')

# 提取训练中评估结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
evaluation = pd.DataFrame(list(zip(epochs, acc, val_acc, loss, val_loss)),
                          columns=['epochs', 'acc', 'val_acc', 'loss', 'val_loss'])

# 绘图
sns.scatterplot(evaluation, x='epochs', y='acc', label='Training')
sns.lineplot(evaluation, x='epochs', y='val_acc', label='Validation', color='darkorange')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('../figure/accuracy.svg', dpi=600, format='svg')
plt.figure()
sns.scatterplot(evaluation, x='epochs', y='loss', label='Training')
sns.lineplot(evaluation, x='epochs', y='val_loss', label='Validation', color='darkorange')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('../figure/loss.svg', dpi=600, format='svg')
plt.show()
