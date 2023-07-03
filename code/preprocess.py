import os
import shutil

# 原始数据集路径
original_train_dir = '../data/original/train'
# 新数据集路径
new_train_dir = '../data/new/train'
new_validation_dir = '../data/new/validation'
new_test_dir = '../data/new/test'
new_dir_lst = [new_train_dir, new_validation_dir, new_test_dir]

for i in range(12500):
    src_cat = os.path.join(original_train_dir, f'cat.{i}.jpg')
    src_dog = os.path.join(original_train_dir, f'dog.{i}.jpg')
    if i < 7500:
        dst_cat = os.path.join(new_train_dir, 'cats', f'cat.{i}.jpg')
        dst_dog = os.path.join(new_train_dir, 'dogs', f'dog.{i}.jpg')
        shutil.copyfile(src_cat, dst_cat)
        shutil.copyfile(src_dog, dst_dog)
    elif i < 10000:
        dst_cat = os.path.join(new_validation_dir, 'cats', f'cat.{i}.jpg')
        dst_dog = os.path.join(new_validation_dir, 'dogs', f'dog.{i}.jpg')
        shutil.copyfile(src_cat, dst_cat)
        shutil.copyfile(src_dog, dst_dog)
    else:
        dst_cat = os.path.join(new_test_dir, 'cats', f'cat.{i}.jpg')
        dst_dog = os.path.join(new_test_dir, 'dogs', f'dog.{i}.jpg')
        shutil.copyfile(src_cat, dst_cat)
        shutil.copyfile(src_dog, dst_dog)


print('total training cat images:', len(os.listdir('../data/new/train/cats')))
print('total training dog images:', len(os.listdir('../data/new/train/dogs')))
print('total validation cat images:', len(os.listdir('../data/new/validation/cats')))
print('total validation dog images:', len(os.listdir('../data/new/validation/dogs')))
print('total test cat images:', len(os.listdir('../data/new/test/cats')))
print('total test dog images:', len(os.listdir('../data/new/test/dogs')))
