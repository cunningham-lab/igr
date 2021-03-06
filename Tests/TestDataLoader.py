import unittest
from matplotlib import pyplot as plt
from Utils.load_data import load_vae_dataset


class MyTestCase(unittest.TestCase):

    def test_data_loading(self):
        dataset_name = 'celeb_a'
        batch_n = 64
        epochs = 100
        run_with_sample = True
        hyper = {'batch_n': batch_n, 'epochs': epochs}
        output = load_vae_dataset(dataset_name=dataset_name, batch_n=batch_n, epochs=epochs,
                                  run_with_sample=run_with_sample, hyper=hyper)
        train_dataset, test_dataset, test_images, hyper = output
        plt.imshow(test_images[0, :, :, :])
        plt.show()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
