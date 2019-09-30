# -*- coding=UTF-8 -*-

import os
import datetime
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader

from config import config
from utils.datasets import ReadCsvImageDataSet
from utils.glaucoma_compiler import GlaucomaClassificationCompiler
from models.unet_classification import UNet_classification


if __name__ == '__main__':
    opt = config
    print(opt)

    # Validation index
    val_idx = 1 # Or your can use config to modify: val_idx = opt.val_index


    # Configure dataloaders
    transforms_val = [
        transforms.Resize((opt.img_crop_height, opt.img_crop_width), Image.BICUBIC),
        transforms.ToTensor(),
    ]

    csv_file_path = opt.dataset_path + "/" + "list.csv"
    img_container = opt.dataset_path

    # Validation dataset read from CSV
    val_dataset = ReadCsvImageDataSet(csv_file_path_=csv_file_path,
                                      transforms_=transforms_val,
                                      image_container_=img_container,
                                      mode='val',
                                      validation_index_=val_idx,
                                      use_grayscale_mask=False)  # True: load mask with grayscale, False: RGB

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    today = datetime.date.today()
    today = "%d%02d%02d" % (today.year, today.month, today.day)
    dataset_name = os.path.split(opt.dataset_path)[-1]

    model = UNet_classification(3, 3) # UNet(input_channels, output_channels)
    model_name = model.__class__.__name__

    # Pre-train model
    load_model_path = r"YOUR_CLASSIFICATION_MODEL_PATH"  # Or your can use config to modify: load_model_path = opt.load_model_path

    # Validation configuration & hyper-parameters
    val_configuration = {
        "validation_index": val_idx,
        "today": today,
        "dataset_name": dataset_name,
        "model": model,
        "model_name": model_name,

        "load_model_path": load_model_path,  # inference

        "validation_dataloader": DataLoader(dataset=val_dataset,
                                            batch_size=opt.val_batch_size,
                                            shuffle=True,
                                            num_workers=opt.n_cpu),  # Thread

    }

    val_compiler = GlaucomaClassificationCompiler(**val_configuration)
    val_compiler.test(is_save_image=True)
