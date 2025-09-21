def crop_img(img, image_size=(224, 224)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        print("Warning: No contours found, returning resized original image.")
        return cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)
    
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    new_img = cv2.resize(new_img, image_size, interpolation=cv2.INTER_CUBIC)
    return new_img

def image_preprocessing(source_dir, saved_root_dir, image_size=(224, 224), channels=3):
    if not os.path.exists(source_dir):
        raise Exception(f"Source directory: {source_dir} does not exist")
    if not os.path.isdir(source_dir):
        raise Exception(f"Source path: {source_dir} is not a directory")

    if not os.path.exists(saved_root_dir):
        os.makedirs(saved_root_dir)
        
    source_dir_path = pathlib.Path(source_dir)
    
    for p in tqdm(source_dir_path.iterdir(), desc="Processing folders"):
        dir_name = str(p).split("/")[-1]
        for fp in p.iterdir():
            filename = str(fp).split("/")[-1]

            img = tf.io.read_file(str(fp))
            img = tf.image.decode_jpeg(img, channels=channels)
            img = crop_img(img.numpy(), image_size)
            img = pil.Image.fromarray(img)

            saved_dist_dir = os.path.join(saved_root_dir, dir_name)
            if not os.path.exists(saved_dist_dir):
                os.makedirs(saved_dist_dir)

            img_dist_path = os.path.join(saved_dist_dir, filename)
            img.save(img_dist_path)
    print(f"\n✅ All images processed and saved to: {saved_root_dir}")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalization
    rotation_range=12,       # Random rotation up to 12 degrees
    zoom_range=0.2,          # Random zoom up to 20%
    horizontal_flip=True     # Random horizontal flip
)

# For validation/testing (only normalization)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators using flow_from_directory (no need for dataframes)
train_gen = train_datagen.flow_from_directory(
    "/kaggle/working/processed/Training",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    "/kaggle/working/processed/Testing",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

classes = list(test_gen.class_indices.keys())
global classes 

# Update your class names from the generator
class_names = list(train_gen.class_indices.keys())
print("Class names:", class_names)
# Update class weight calculation using generator classes
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Update your visualization code
for images, labels in train_gen:
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")
    break  # Just show first batch