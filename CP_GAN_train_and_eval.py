import Input_and_Utils
import os
import tensorflow as tf
import time
import numpy as np

from CPGAN_model import *
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import CSVLogger
from keras.losses import binary_crossentropy
from keras import backend as K


# Path for Discriminator and Generator weights of the original architecture
DISC_MODEL_PATH = '../disc_weight.h5'
GEN_MODEL_PATH = '../gen_weight.h5'
# Path for Generator weights of the U-NET architecture
UNET_GEN_MODEL_PATH = '../unet_gen_weight.h5'
# Path for the weights of the supervised network
SUPERVISED_GEN_PATH = '../super_weight.h5'
# Input shapes, careful!! the proportions have to be in check to work
WIDTH_CROP = 128
HEIGHT_CROP = 128
WIDTH_ORG = 144    # WIDTH_CROP + PADDING in CPGAN_model
HEIGHT_ORG = 144    # HEIGHT_CROP + PADDING in CPGAN_model
WIDTH_REAL = 72     # Half the size of WIDTH_ORG
HEIGHT_REAL = 72   # Half the size of HEIGHT_ORG
CHANNELS = 3
# Some parameters for fine-tuning and options for training
BATCH_SIZE = 4
DISC_ITER = 1
GAN_ITER = 1
CYCLES = 1002
NOISE = False
BLUR = False
LEARNING_RATE = 0.0001
# Available Optimizers 'adam', 'sgd', 'RMSprop'
CPGAN_OPTIMIZER = 'adam'
DISC_OPTIMIZER = 'adam'
SUPER_OPTIMIZER = 'adam'
PSEUDO_OPTIMIZER = 'adam'
# Some options concerning modes and computational sources
PRETRAINED = True
MODEL = 'paper_cpgan'  # 'unet_cpgan', 'paper_cpgan' or 'supervised'
SUPERVISED_MODEL = 'paper'  # 'paper' or 'unet
UNIT = 'CPU'  # 'GPU' or 'CPU'
NUM_CORES = 4
MODE = 'train'  # Mode 'train' or 'evaluate'
# Category to determine which images to use
CATEGORY = 'person'
# CSV Logger for extracting the results to a external csv file
GEN_LOGGER = CSVLogger(filename='../gen_logger.csv',
                       separator=';',
                       append=True)
DISC_LOGGER = CSVLogger(filename='../disc_logger.csv',
                        separator=';',
                        append=True)
SUPER_LOGGER = CSVLogger(filename='../super_logger.csv',
                         separator=';',
                         append=True)


# Dictionary of available optimizers, selection at the top of the page
# Adjust learning rate at the top of the page
def get_optim(optim):
    return {
        'sgd': SGD(lr=LEARNING_RATE, momentum=0, decay=0, nesterov=False),
        'adam': Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=0.0000001, decay=0, amsgrad=False),
        'RMSprop': RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)
    }.get(optim, Adam(lr=LEARNING_RATE))


# CPGAN class to create a GAN with 'mse' as loss function
class Paper_CPGAN:

    def __init__(self):

        if MODEL == 'unet_cpgan':
            self.model_gen = unet_gen_for_gan()
            GEN_PATH = UNET_GEN_MODEL_PATH
            DISC_PATH = UNET_DISC_MODEL_PATH
        else:
            self.model_gen = paper_gen()
            GEN_PATH = GEN_MODEL_PATH
            DISC_PATH = DISC_MODEL_PATH

        # Load Discriminator
        self.model_disc = paper_disc()
        if os.path.isfile(DISC_PATH) and PRETRAINED:
            print('loading weights for discriminator...')
            self.model_disc.load_weights(DISC_PATH)

        # Load Generator
        if os.path.isfile(GEN_PATH) and PRETRAINED:
            print('loading weights for generator...')
            self.model_gen.load_weights(GEN_PATH)

        # Construct and compile GAN
        org_crop = Input(shape=(WIDTH_CROP, HEIGHT_CROP, CHANNELS))
        background = Input(shape=(WIDTH_REAL, HEIGHT_REAL, CHANNELS))

        pred = self.model_gen([background, org_crop])
        fake = pred[0]
        mask = pred[1]

        self.model_disc.trainable = False
        self.model_gen.trainable = True
        res_fake = self.model_disc(fake)

        self.cp_gan = Model(inputs=[background, org_crop], outputs=[res_fake, mask], name='CPGAN')
        self.cp_gan.compile(optimizer=get_optim(CPGAN_OPTIMIZER),
                            loss={'Disc': 'mse', 'Gen': mask_loss},
                            loss_weights={'Disc': 1.0, 'Gen': 1.0},
                            metrics={'Disc': 'accuracy', 'Gen': [conf_metric, min_metric]}
                            )

        self.model_disc.trainable = True

        # Compile Discriminator
        self.model_disc.compile(optimizer=get_optim(DISC_OPTIMIZER),
                                loss=['mse'],
                                loss_weights=[1],
                                metrics=['accuracy'])

    # Train CPGAN with 'mse' as a loss function
    def train_paper_cpgan(self):

        if MODEL == 'unet_cpgan':
            GEN_PATH = UNET_GEN_MODEL_PATH
            DISC_PATH = UNET_DISC_MODEL_PATH
        else:
            GEN_PATH = GEN_MODEL_PATH
            DISC_PATH = DISC_MODEL_PATH

        dataset = Input_and_Utils.get_train_dataset()
        gan_train = 0
        disc_train = 0

        start_time = time.time()
        print('Start training....')

        for cycle in range(CYCLES):
            print('Cycle %i | %i' % (cycle, CYCLES))

            # Prepare the input data
            images, anns = Input_and_Utils.get_image_and_anns(dataset=dataset, batch_size=6, category=CATEGORY,
                                                              noise=NOISE, blur=BLUR)

            org_images, crop_list, bboxes = Input_and_Utils.get_cropped_images(images=images, anns=anns,
                                                                               shape_0=WIDTH_CROP, shape_1=HEIGHT_CROP,
                                                                               custom_size=True)

            temp1, real_list, temp2 = Input_and_Utils.get_cropped_images(images=images, anns=anns,
                                                                         shape_0=WIDTH_ORG, shape_1=HEIGHT_ORG,
                                                                         custom_size=True)

            real_list = Input_and_Utils.resize_images(real_list, WIDTH_REAL, HEIGHT_REAL)

            temp1, backgrounds, temp2 = Input_and_Utils.get_cropped_images(images=org_images,
                                                                           shape_0=WIDTH_ORG, shape_1=HEIGHT_ORG,
                                                                           random_position=True,
                                                                           custom_size=True)

            backgrounds = Input_and_Utils.resize_images(backgrounds, WIDTH_REAL, HEIGHT_REAL)

            backgrounds = backgrounds[0:BATCH_SIZE]
            crop_list = crop_list[0:BATCH_SIZE]

            if len(backgrounds) < BATCH_SIZE or len(crop_list) < BATCH_SIZE:
                continue

            real = np.ones((BATCH_SIZE, 1))
            real_list = np.array(real_list[0:BATCH_SIZE])

            placeholder_mask = np.ones((BATCH_SIZE, 64, 64))

            backgrounds = np.array(backgrounds)
            crop_list = np.array(crop_list)

            fake = np.zeros((BATCH_SIZE, 1))

            predictions = self.model_gen.predict([backgrounds, crop_list],
                                                 verbose=0,
                                                 steps=None)

            if (cycle % 50) == 0:
                Input_and_Utils.save_image((predictions[0][0] + 1) / 2, 'cycle %s' % cycle)

            # Train GAN
            print('Generator training.....')
            self.cp_gan.fit(x=[backgrounds, crop_list],
                            y=[real, placeholder_mask],
                            batch_size=BATCH_SIZE, epochs=GAN_ITER,
                            verbose=1, callbacks=[GEN_LOGGER])

            gan_train += len(crop_list)
            print('GAN trained on %s images' % gan_train)

            # Train Discriminator
            print('Discriminator training.....')

            self.model_disc.fit(x=predictions[0], y=fake,
                                batch_size=BATCH_SIZE, epochs=DISC_ITER,
                                verbose=1, callbacks=[DISC_LOGGER])

            self.model_disc.fit(real_list, y=real,
                                batch_size=BATCH_SIZE, epochs=DISC_ITER,
                                verbose=1, callbacks=[DISC_LOGGER])

            disc_train += len(fake)
            print('Discriminator trained on %s images' % disc_train)

            if (cycle % 25) == 0:
                self.model_disc.save_weights(DISC_PATH)
                self.model_gen.save_weights(GEN_PATH)
                print('save weights....')

        self.model_disc.save_weights(DISC_PATH)
        self.model_gen.save_weights(GEN_PATH)
        print('save weights....')
        print('training finished in %s' % (time.time() - start_time))


# Train generator with ground truth
def train_with_gt(model='paper'):

    dataset = Input_and_Utils.get_train_dataset()

    if model == 'paper':
        model_gen = super_paper_gen()
        if os.path.isfile(SUPERVISED_GEN_PATH) and PRETRAINED:
            print('loading weights for generator...')
            model_gen.load_weights(SUPERVISED_GEN_PATH)

    elif model == 'unet':
        model_gen = super_unet_gen()
        if os.path.isfile(SUPERVISED_GEN_PATH) and PRETRAINED:
            print('loading weights for generator...')
            model_gen.load_weights(SUPERVISED_GEN_PATH)

    else:
        print('no model available')
        exit()

    model_gen.compile(optimizer=get_optim(SUPER_OPTIMIZER),
                      loss=[binary_crossentropy, mask_loss],
                      loss_weights=[1, 1],
                      metrics=['accuracy'])

    number_img = 0
    start_time = time.time()
    print('Start training....')

    for cycle in range(CYCLES):
        print('Cycle %i | %i' % (cycle, CYCLES))

        images, anns = Input_and_Utils.get_image_and_anns(dataset=dataset, batch_size=2*BATCH_SIZE, category=CATEGORY)
        org_images, crop_list, bboxes = Input_and_Utils.get_cropped_images(images=images, anns=anns,
                                                                           shape_0=WIDTH_CROP, shape_1=HEIGHT_CROP,
                                                                           custom_size=True)

        gt_masks = []
        for ann in anns:
            for seg in ann:
                if seg['iscrowd'] == 1 or seg['area'] < 1500 or seg['area'] > 17000:
                    continue
                gt_mask = dataset.annToMask(seg)
                gt_masks.append(gt_mask)

        gt = Input_and_Utils.get_cropped_eval(masks=gt_masks, anns=anns, shape_0=WIDTH_CROP, shape_1=HEIGHT_CROP)
        gt = Input_and_Utils.resize_images(gt, 64, 64)

        if len(crop_list) == 0:
            continue

        if (cycle % 30) == 0:
            Input_and_Utils.save_image(model_gen.predict(np.expand_dims(crop_list[0], axis=0))[0],
                                       'super_mask %i' % cycle)
            Input_and_Utils.save_image(gt[0], 'gt %i' % cycle)

        Input_and_Utils.save_image(gt[0], 'sup')
        Input_and_Utils.save_image((crop_list[0]+1)/2, 'sup2')
        gt = gt[0:len(crop_list)]
        crop_list = np.array(crop_list)
        gt = np.array(gt)

        model_gen.fit(x=crop_list, y=[gt, gt],
                      batch_size=BATCH_SIZE, epochs=GAN_ITER,
                      verbose=1, callbacks=[SUPER_LOGGER])

        number_img += len(crop_list)
        print('trained on %s images' % number_img)

        if (cycle % 30) == 0:
            model_gen.save_weights(SUPERVISED_GEN_PATH)
            print('save weights....')

    model_gen.save_weights(SUPERVISED_GEN_PATH)
    print('save weights....')
    print('training finished in %s' % (time.time() - start_time))


# Evaluate the performence of a model by comparing predicted mask with
# the ground truth and compute IoU
def evaluate():

    dataset = Input_and_Utils.get_val_dataset()

    if MODEL == 'pseudo':
        print('load pseudo....')
        model_gen = super_unet_gen()
        model_gen.load_weights(PSEUDO_SUPERVISED_GEN_PATH)

    elif MODEL == 'supervised':
        print('load supervised...')
        model_gen = super_paper_gen()
        model_gen.load_weights(SUPERVISED_GEN_PATH)

    elif MODEL == 'unet_cpgan':
        print('load UNET_GAN...')
        model_disc = paper_disc()
        model_disc.load_weights(UNET_DISC_MODEL_PATH)
        model_gen = unet_gen_for_gan()
        model_gen.load_weights(UNET_GEN_MODEL_PATH)

    elif MODEL == 'paper_cpgan':
        print('load CPGAN...')
        model_disc = paper_disc()
        model_disc.load_weights(DISC_MODEL_PATH)
        model_gen = paper_gen()
        model_gen.load_weights(GEN_MODEL_PATH)

    else:
        print('no such model for evaluation available')
        exit()

    images, anns = Input_and_Utils.get_image_and_anns(dataset=dataset, batch_size=15, category=CATEGORY)
    org_images, cropped_images, bboxes = Input_and_Utils.get_cropped_images(images, anns,
                                                                            shape_0=WIDTH_CROP, shape_1=HEIGHT_CROP,
                                                                            custom_size=True)

    dt_masks = []
    gt_masks = []

    for ann in anns:
        for seg in ann:
            if seg['iscrowd'] == 1 or seg['area'] < 1500 or seg['area'] > 17000:
                continue
            gt_mask = dataset.annToMask(seg)
            gt_masks.append(gt_mask)

    gt_masks = Input_and_Utils.get_cropped_eval(gt_masks, anns, shape_0=WIDTH_CROP, shape_1=HEIGHT_CROP)

    gts = Input_and_Utils.resize_images(gt_masks, 64, 64)

    if MODEL in ['unet_cpgan', 'paper_cpgan']:

        temp1, real_list, temp2 = Input_and_Utils.get_cropped_images(images=images, anns=anns,
                                                                     shape_0=WIDTH_ORG, shape_1=HEIGHT_ORG,
                                                                     custom_size=True)

        real_list = Input_and_Utils.resize_images(real_list, WIDTH_REAL, HEIGHT_REAL)

        temp1, backgrounds, temp2 = Input_and_Utils.get_cropped_images(images=org_images,
                                                                       shape_0=WIDTH_ORG, shape_1=HEIGHT_ORG,
                                                                       random_position=True,
                                                                       custom_size=True)
        backgrounds = Input_and_Utils.resize_images(backgrounds, WIDTH_REAL, HEIGHT_REAL)

        res_real = 0
        res_fake = 0
        i = 0

        for background, crop in zip(backgrounds, cropped_images):

            pred = model_gen.predict([np.expand_dims(background, axis=0), np.expand_dims(crop, axis=0)])

            dt_mask = np.squeeze(np.round(pred[1]), axis=3)
            dt_masks.append(dt_mask)

            res_real += model_disc.predict(np.expand_dims(real_list[i], axis=0))
            res_fake += model_disc.predict(pred[0])

            if i % 1 == 0:
                Input_and_Utils.save_image((pred[0]+1)/2, 'cp_img %s' % i, eval_directory=True)

            i += 1

        print('real_avg: %s  fake_avg: %s' % (res_real/len(backgrounds), res_fake/len(backgrounds)))

    else:  # 'supervised', 'pseudo'

        for crop in cropped_images:
            pred = model_gen.predict(np.expand_dims(crop, axis=0))[0]
            dt_masks.append(np.squeeze(pred, axis=3))

    # computation intersection over union
    avg = 0
    i = 0
    n = 0
    crop = Input_and_Utils.resize_images(cropped_images, 64, 64)

    for dt, gt in zip(dt_masks, gts):
        dt = dt.astype(bool)
        gt = gt.astype(bool)
        if np.sum(gt) == 0:
            continue
        overlap = dt * gt
        union = dt + gt
        IoU = np.sum(overlap)/(np.sum(union))
        avg += IoU
        n += 1

        if i % 1 == 0:
            Input_and_Utils.save_image((cropped_images[i]+1)/2, 'org %i' % i, eval_directory=True)
            Input_and_Utils.save_image(((crop[i]+1)/2) * np.expand_dims(np.squeeze(dt, axis=0), axis=2),
                                       'seg %i' % i, eval_directory=True)
            Input_and_Utils.save_image(dt, 'dt %i' % i, eval_directory=True)
            Input_and_Utils.save_image(gt, 'gt %i' % i, eval_directory=True)
        i += 1

    print('Average IoU:', avg / n)


if __name__ == '__main__':

    if UNIT == 'GPU':
        num_GPU = 1
        num_CPU = 1
    elif UNIT == 'CPU':
        num_CPU = 1
        num_GPU = 0
    else:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_CORES,
                            inter_op_parallelism_threads=NUM_CORES,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )

    session = tf.Session(config=config)
    K.set_session(session)

    # training_mode
    if MODE == 'train':

        if MODEL == 'unet_cpgan':
            cpgan = Paper_CPGAN()
            cpgan.train_paper_cpgan()

        elif MODEL == 'paper_cpgan':
            cpgan = Paper_CPGAN()
            cpgan.train_paper_cpgan()

        elif MODEL == 'supervised':
            train_with_gt(model=SUPERVISED_MODEL)

        else:
            print('no such model available')

    # evaluation mode
    elif MODE == 'evaluate':

        evaluate()

    else:
        print('please select a valid mode')
