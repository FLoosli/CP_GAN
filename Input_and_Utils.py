import skimage.io
import numpy as np
import skimage

from skimage.transform import resize
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy import ndimage


# Directory to which the images are saved to
SAVE_DIRECTORY = '../Examples_Results'
EVAL_DIRECTORY = '../Evaluation'
# Dataset paths
VAL_PATH = '../instances_val2017.json'
TRAIN_PATH = '../instances_train2017.json'


# cut and paste function to cutout a mask from a image and paste at another position
# image: Original image for acquiring new background position
# image_cropped: Cut out mask for pasting
# mask: Calculated mask for detected object
def cut_and_paste(org_image, crop_image, mask):

    mask = np.squeeze(mask, axis=0)

    crop_image = resize_images(crop_image, mask.shape[0], mask.shape[1])[0]

    mask_to_paste = crop_image * np.expand_dims(mask, axis=2)
    mask_to_paste = np.pad(mask_to_paste, ((8, 0), (4, 4), (0, 0)),  mode='constant', constant_values=0)

    inverted_mask = 1 - mask
    inverted_mask = np.pad(inverted_mask, ((8, 0), (4, 4)),  mode='constant', constant_values=0)

    temp1, img_to_paste, temp2 = get_cropped_images([org_image], random_position=True,
                                                    shape_0=mask_to_paste.shape[0],
                                                    shape_1=mask_to_paste.shape[1],
                                                    custom_size=True)

    cp_image = (img_to_paste[0] * np.expand_dims(inverted_mask, axis=2)) + mask_to_paste

    return cp_image


# resize image to a desired size with skimage function
# returns list of resized images
def resize_images(images, dim_0, dim_1):

    images_resized = []

    for image in images:
        if image.ndim == 4:
            image = np.squeeze(image, axis=0)

        if image.ndim == 3:
            if image.shape[2] == 1:
                img = skimage.transform.resize(image, (dim_0, dim_1, 1), mode='constant', anti_aliasing=True)

            elif image.shape[2] == 3:
                img = skimage.transform.resize(image, (dim_0, dim_1, 3), mode='constant', anti_aliasing=True)

        else:
            img = skimage.transform.resize(image, (dim_0, dim_1), mode='constant', anti_aliasing=True)
            img = np.expand_dims(img, axis=2)

        images_resized.append(img)

    return images_resized


# pad or slice a picture to the desired size dependent on the input size
# returns list of resized images
def cut_to_size(images, desired_x, desired_y):

    sized_images = []

    for image in images:

        corr_x = 0
        corr_y = 0

        if image.ndim == 3:

            if desired_x > image.shape[0]:
                pad_x = int((desired_x - image.shape[0]) * 0.5)
                if 2 * pad_x + image.shape[0] != desired_x:
                    corr_x += 1
                image = np.pad(image, ((2*pad_x+corr_x, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            else:
                start_x = int(image.shape[0]/2)
                cut_x = int(desired_x/2)
                if cut_x*2 != desired_x:
                    corr_x += 1
                image = image[start_x-cut_x-corr_x:start_x+cut_x+corr_x, :, :]

            if desired_y > image.shape[1]:
                pad_y = int((desired_y - image.shape[1]) * 0.5)
                if 2 * pad_y + image.shape[1] != desired_y:
                    corr_y += 1
                image = np.pad(image, ((0, 0), (pad_y+corr_y, pad_y), (0, 0)), mode='constant', constant_values=0)
            else:
                start_y = int(image.shape[1] / 2)
                cut_y = int(desired_y / 2)
                if cut_y * 2 != desired_y:
                    corr_y += 1
                image = image[:, start_y-cut_y-corr_y:start_y+cut_y+corr_y, :]

        else:

            if desired_x > image.shape[0]:
                pad_x = int((desired_x - image.shape[0]) * 0.5)
                if 2 * pad_x + image.shape[0] != desired_x:
                    corr_x += 1
                image = np.pad(image, ((2*pad_x+corr_x, 0), (0, 0)), mode='constant', constant_values=0)
            else:
                start_x = int(image.shape[0] / 2)
                cut_x = int(desired_x / 2)
                if cut_x * 2 != desired_x:
                    corr_x += 1
                image = image[start_x - cut_x:start_x + cut_x + corr_x, :]

            if desired_y > image.shape[1]:
                pad_y = int((desired_y - image.shape[1]) * 0.5)
                if 2 * pad_y + image.shape[1] != desired_y:
                    corr_y += 1
                image = np.pad(image, ((0, 0), (pad_y+corr_y, pad_y)), mode='constant', constant_values=0)
            else:
                start_y = int(image.shape[1] / 2)
                cut_y = int(desired_y / 2)
                if cut_y * 2 != desired_y:
                    corr_y += 1
                image = image[:, start_y - cut_y:start_y + cut_y + corr_y]

        sized_images.append(image)

    return sized_images


# crop image according to the given bounding box or arbitrary coordinates
# restriction on size of the bounding box and checks if its crowded
# returns list of cropped parts of the input image, the original images and the coordinates of the bounding boxes
def get_cropped_images(images, anns=None, pos_x=None, pos_y=None,
                       shape_0=None, shape_1=None,
                       random_position=True, custom_size=False):

    org_images = []
    cropped_images = []
    bboxes = []

    if (shape_0 is not None) and (shape_1 is not None) and (anns is None):
        for image in images:

            bbox = []
            corr_x = 0
            corr_y = 0

            if random_position:

                if image.shape[0] < shape_0:
                    continue
                else:
                    pos_x = np.random.randint(shape_0/2, image.shape[0] - shape_0/2 + 1)

                if image.shape[1] < shape_1:
                    continue
                else:
                    pos_y = np.random.randint(shape_1/2, image.shape[1] - shape_1/2 + 1)

                dim_0 = int(shape_0/2)
                dim_1 = int(shape_1/2)

                if dim_0 * 2 != shape_0:
                    corr_x = 1

                if dim_1 * 2 != shape_1:
                    corr_y = 1

                random_crop = image[pos_x-dim_0+corr_x:pos_x + dim_0,
                                    pos_y-dim_1+corr_y:pos_y + dim_1,
                                    ]

            else:
                if (pos_x is None) or (pos_y is None):
                    print('no enough coordinates given pos_x or pos_y missing')
                    return images, cropped_images, bboxes

                else:
                    dim_0 = int(shape_0)
                    dim_1 = int(shape_1)

                    random_crop = image[pos_x:pos_x + dim_0,
                                        pos_y:pos_y + dim_1,
                                        ]

                    if (pos_x - dim_0 < 0) or (pos_y - dim_1 < 0):
                        random_crop = cut_to_size([random_crop], dim_0, dim_1)[0]

            if random_crop.ndim == 2:
                random_crop = np.stack((random_crop,)*3, axis=-1)

            org_images.append(image)
            cropped_images.append(random_crop)
            bbox.extend([pos_x, pos_y, dim_0, dim_1])
            bboxes.append(bbox)

    else:
        for i, seg in zip(range(len(images)), anns):
            for ann in seg:
                image = images[i]
                if ann['iscrowd'] == 1 or ann['area'] < 1500 or ann['area'] > 17000 or image.ndim == 2:
                    continue

                else:
                    bbox = ann['bbox']
                    crop_y = int(bbox[0])
                    crop_x = int(bbox[1])
                    height = int(bbox[2]/2)
                    width = int(bbox[3]/2)

                    if custom_size:
                        crop_width = int(shape_0/2)
                        crop_height = int(shape_1/2)
                    else:
                        crop_height = int(bbox[2]/2)
                        crop_width = int(bbox[3]/2)

                    start_x = crop_x + width
                    start_y = crop_y + height

                    if start_x - crop_width < 0:
                        crop_width = start_x
                    if start_y - crop_height < 0:
                        crop_height = start_y

                    if start_x + crop_width > image.shape[0]:
                        crop_width = image.shape[0] - start_x
                    if start_y + crop_height > image.shape[1]:
                        crop_height = image.shape[1] - start_y

                    if image.shape == 1 or image.shape == 0 or crop_width*2 > shape_0 or crop_height*2 > shape_1:
                        continue

                    elif len(image.shape) == 2:
                        img_cropped = image[start_x - crop_width:start_x + crop_width,
                                            start_y - crop_height:start_y + crop_height]
                        img_cropped = np.expand_dims(img_cropped, axis=2)
                    else:
                        img_cropped = image[start_x - crop_width:start_x + crop_width,
                                            start_y - crop_height:start_y + crop_height,
                                            :]

                    if img_cropped.shape[0] != shape_0 or img_cropped.shape[1] != shape_1:
                        img_cropped = cut_to_size([img_cropped], shape_0, shape_1)[0]

                    org_images.append(image)
                    cropped_images.append(img_cropped)
                    bboxes.append(bbox)

    return org_images, cropped_images, bboxes


# crop function for evaluation
# used to crop the ground truth masks correctly
# returns the cropped ground truth masks
def get_cropped_eval(masks, anns, shape_0, shape_1):

    cropped_masks = []

    i = 0
    for seg in anns:
        for ann in seg:
            if ann['iscrowd'] == 1 or ann['area'] < 1500 or ann['area'] > 17000:
                continue

            else:
                mask = masks[i]
                bbox = ann['bbox']
                crop_y = int(bbox[0])
                crop_x = int(bbox[1])
                height = int(bbox[2]/2)
                width = int(bbox[3]/2)

                crop_width = int(shape_0/2)
                crop_height = int(shape_1/2)

                start_x = crop_x + width
                start_y = crop_y + height

                if start_x - crop_width < 0:
                    crop_width = start_x
                if start_y - crop_height < 0:
                    crop_height = start_y

                if start_x + crop_width > mask.shape[0]:
                    crop_width = mask.shape[0] - start_x
                if start_y + crop_height > mask.shape[1]:
                    crop_height = mask.shape[1] - start_y

                if mask.shape == 1 or mask.shape == 0 or crop_width * 2 > shape_0 or crop_height * 2 > shape_1:
                    continue

                mask_cropped = mask[start_x - crop_width:start_x + crop_width,
                                    start_y - crop_height:start_y + crop_height]

                if mask_cropped.shape[0] != shape_0 or mask_cropped.shape[1] != shape_1:
                    mask_cropped = cut_to_size([mask_cropped], shape_0, shape_1)[0]

                cropped_masks.append(mask_cropped)
                i += 1

    return cropped_masks


# Get all relevant images for training according to category
# set category at the top of the file
def get_image_and_anns(dataset, batch_size, category, noise=False, blur=False):

    images = []
    image_anns = []

    cat_ids = dataset.getCatIds(catNms=[category])
    img_ids = dataset.getImgIds(catIds=cat_ids)

    for i in range(batch_size):

        img = dataset.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]
        image = (skimage.io.imread(img['coco_url'], as_gray=False)/127.5) - 1

        ann_ids = dataset.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = dataset.loadAnns(ids=ann_ids)

        if np.any(noise):
            noise = np.random.normal(0, 1, image.shape)
            noise = noise.reshape(image.shape)
            image = image + 0.5*noise

        if np.any(blur):
            image = ndimage.gaussian_filter(image, sigma=2)

        images.append(image)
        image_anns.append(anns)

    return images, image_anns


# save picture to SAVE_DIRECTORY
# set the path at the top of the page
def save_image(image, file_name='example', eval_directory=False):

    if image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=2)

    plt.imshow(image)
    plt.axis('off')
    if eval_directory:
        plt.savefig(EVAL_DIRECTORY + file_name)
    else:
        plt.savefig(SAVE_DIRECTORY + file_name)
    print('%s saved' % file_name)


# Get usable data_annotations from the val_dataset of COCO, saved at VAL_PATH
# Path variable at the top of the page
def get_val_dataset():
    val_coco = COCO(VAL_PATH)
    return val_coco


# Get usable data_annotations from the train_dataset of COCO, saved at TRAIN_PATH
# Path variable at the top of the page
def get_train_dataset():
    train_coco = COCO(TRAIN_PATH)
    return train_coco
