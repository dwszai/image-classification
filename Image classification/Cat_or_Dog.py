
#=======================================================================================
# ==============================RUN BUILT MODEL=========================================
#=======================================================================================

cat_count = 0
dog_count = 0
# Saving/ Loading model for future uses
from keras.models import load_model
# classifier.save('model.m1')
classifier = load_model('CNN_Model.m1')

# Predict image using trained model (either this or above)
import numpy as np
from keras.preprocessing import image

inpu = 1
while inpu != 0:
	filename = input('Enter the file name: ')
	test_image = image.load_img('single_prediction/' + filename, target_size = (64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict_classes(test_image)

	if result[0][0] == 1:
	    prediction = 'dog'
	    print('This image contains a DOG.')
	    dog_count += 1
	else:
	    prediction = 'cat'
	    print('This image contains a CAT.')
	    cat_count += 1

	print(f'Number of dog: {dog_count}')
	print(f'Number of cat: {cat_count}')