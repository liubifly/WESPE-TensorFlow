from WESPE_DIV2K import *
from dataloader.dataloader_DIV2K import *
from ops import *
from utils import *
from easydict import EasyDict as edict


def main():
    config = edict()
    # training parameters
    config.batch_size = 32  # 32
    config.patch_size = 100
    config.mode = "RGB"
    config.channels = 3
    config.content_layer = 'relu2_2'  # originally relu5_4 in DPED
    config.learning_rate = 1e-4
    config.augmentation = True  # data augmentation (flip, rotation)
    config.test_every = 500
    config.train_iter = 10000

    # weights for loss
    config.w_content = 0.2  # reconstruction (originally 1)
    config.w_color = 40  # gan color (originally 5e-3)
    config.w_texture = 3  # gan texture (originally 5e-3)
    config.w_tv = 1 / 400  # total variation (originally 400)

    config.model_name = "WESPE_DIV2K"

    config.dataset_name = "sony"
    config.train_path_phone = os.path.join("/home/ubuntu/dped", str(config.dataset_name), "training_data",
                                           str(config.dataset_name), "*.jpg")
    config.train_path_DIV2K = os.path.join("/home/ubuntu/DIV2K/DIV2K_train_HR/*.png")

    config.test_path_phone_patch = os.path.join("/home/ubuntu/dped", str(config.dataset_name),
                                                "test_data/patches", str(config.dataset_name), "*.jpg")
    config.test_path_phone_image = os.path.join(
        "/home/ubuntu/sample_images/original_images",
        str(config.dataset_name), "*.jpg")

    config.vgg_dir = "imagenet-vgg-verydeep-19.mat"

    config.result_dir = os.path.join("./result", config.model_name)
    config.result_img_dir = os.path.join(config.result_dir, "samples")
    config.checkpoint_dir = os.path.join(config.result_dir, "model")

    if not os.path.exists(config.checkpoint_dir):
        print("creating dir...", config.checkpoint_dir)
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.result_dir):
        print("creating dir...", config.result_dir)
        os.makedirs(config.result_dir)
    if not os.path.exists(config.result_img_dir):
        print("creating dir...", config.result_img_dir)
        os.makedirs(config.result_img_dir)

    config.sample_dir = "samples_DIV2K"
    if not os.path.exists(config.sample_dir):
        print("creating dir...", config.sample_dir)
        os.makedirs(config.sample_dir)
    # directories
    # load dataset
    dataset_phone, dataset_DIV2K = load_dataset(config)
    phone_batch, DIV2K_batch = get_batch(dataset_phone, dataset_DIV2K, config, start=0)
    print('done!')
    # build WESPE model
    tf.reset_default_graph()
    # uncomment this when only trying to test the model
    # dataset_phone = []
    # dataset_DIV2K = []
    sess = tf.Session()
    model = WESPE(sess, config, dataset_phone, dataset_DIV2K)
    # train generator & discriminator together
    model.train(load=True)
    # test trained model
    model.test_generator(200, 4, load=False)
    # save trained model
    model.save()


if __name__ == '__main__':
    main()
