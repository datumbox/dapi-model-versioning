from torchvision import transforms as T


class ImageNetEval:

    def __init__(self, crop_size, resize_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 interpolation=T.InterpolationMode.BILINEAR):

        self.transforms = T.Compose([
            T.Resize(resize_size, interpolation=interpolation),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)
