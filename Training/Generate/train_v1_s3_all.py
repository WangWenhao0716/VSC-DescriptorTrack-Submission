from PIL import Image
from augly.image.transforms import *
import torchvision.transforms as transforms
import random
import augly.image as imaugs
import argparse
import numpy as np

class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")
    
class RandomResizeCrop:
    def __call__(self, x):
        tran = transforms.RandomResizedCrop(256, scale=(0.3,1))
        return tran(x)

class Random_image_opaque:
    def __init__(self, opa = [0.2, 1.0], path = '/raid/VSC/data/training_images_9/', which = [0,100000]):
        self.opa = opa
        self.path = path
        self.which = which
        self.imgs = os.listdir(self.path)

    def __call__(self, x):
        opa = random.uniform(self.opa[0], self.opa[1])
        which = random.randint(self.which[0], self.which[1])
        base = Image.open(self.path+self.imgs[which])
        x = x.resize((256,256))
        base = base.resize((256,256))
        assert base.size == x.size
        x = imaugs.overlay_image(base, x, x_pos=0, y_pos=0, opacity=opa)#0.2~1
        return x
    
class GrayScale:
    def __call__(self, x):
        x = Grayscale()(x)
        return x
    
class Color_jitter():
    def __call__(self, x):
        x = transforms.ColorJitter(brightness=3, contrast=4, saturation=4, hue=0.5)(x)
        return x
    
class RandomBlur:
    def __init__(self, radius = [2, 5]):
        self.radius = radius
    
    def __call__(self, x):
        radius = random.uniform(self.radius[0], self.radius[1])
        x = Blur(radius = radius)(x)
        return x
    
class RandomPixelization:
    def __init__(self, ratios = [0.1, 1]):
        self.ratios = ratios
    
    def __call__(self, x):
        ratio = random.uniform(self.ratios[0], self.ratios[1])
        x = Pixelization(ratio = ratio)(x)
        return x
    
class RandomRotate:
    def __init__(self, degrees = [0,360]):
        self.degrees = degrees
    
    def __call__(self, x):
        degree = random.uniform(self.degrees[0], self.degrees[1])
        x = Rotate(degrees = degree)(x)
        return x
    
class RandomPad:
    def __init__(self, w_factors = [0, 0.5], h_factors = [0, 0.5], color_1s = [0,255], color_2s = [0,255], color_3s = [0,255]):
        self.w_factors = w_factors
        self.h_factors = h_factors
        self.color_1s = color_1s
        self.color_2s = color_2s
        self.color_3s = color_3s
    
    def __call__(self, x):
        w_factor = random.uniform(self.w_factors[0], self.w_factors[1])
        h_factor = random.uniform(self.h_factors[0], self.h_factors[1])
        color_1 = random.randint(self.color_1s[0], self.color_1s[1])
        color_2 = random.randint(self.color_2s[0], self.color_2s[1])
        color_3 = random.randint(self.color_3s[0], self.color_3s[1])
        x = Pad(w_factor = w_factor, h_factor = h_factor, color = (color_1, color_2, color_3))(x)
        return x.resize((256,256))

class RandomAddNoise:
    def __init__(self, means = [0, 0.5], varrs = [0, 0.5]):
        self.means = means
        self.varrs = varrs
    def __call__(self, x):
        mean = random.uniform(self.means[0], self.means[1])
        var = random.uniform(self.varrs[0], self.varrs[1])
        x = RandomNoise(mean = mean, var = var)(x)
        return x
    
class VertFlip:
    def __call__(self, x):
        return VFlip()(x)
    
class HoriFlip:
    def __call__(self, x):
        return HFlip()(x)

class RandomMemeFormat:
    def __init__(self, text_len = [1, 10], path = '/raid/VSC/data/fonts/', opacity = [0, 1], \
                text_colors_0 = [0, 255], text_colors_1 = [0, 255], text_colors_2 = [0, 255], \
                caption_height = [100, 300], \
                bg_colors_0 = [0, 255], bg_colors_1 = [0, 255], bg_colors_2 = [0, 255]):
        self.text_len = text_len
        self.path = path
        self.opacity = opacity
        self.text_colors_0 = text_colors_0
        self.text_colors_1 = text_colors_1
        self.text_colors_2 = text_colors_2
        self.caption_height = caption_height
        self.bg_colors_0 = bg_colors_0
        self.bg_colors_1 = bg_colors_1
        self.bg_colors_2 = bg_colors_2
    
    def __call__(self, x):
        string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        length = random.randint(self.text_len[0], self.text_len[1])
        text = ''.join(random.sample(string, length))
        tiff_path = self.path + random.choice(os.listdir(self.path))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        text_color_0 = random.randint(self.text_colors_0[0], self.text_colors_0[1])
        text_color_1 = random.randint(self.text_colors_1[0], self.text_colors_1[1])
        text_color_2 = random.randint(self.text_colors_2[0], self.text_colors_2[1])
        height = random.randint(self.caption_height[0], self.caption_height[1])
        bg_color_0 = random.randint(self.bg_colors_0[0], self.bg_colors_0[1])
        bg_color_1 = random.randint(self.bg_colors_1[0], self.bg_colors_1[1])
        bg_color_2 = random.randint(self.bg_colors_2[0], self.bg_colors_2[1])
        x = MemeFormat(text = text,
                       font_file = tiff_path,
                       opacity = opacity,
                       text_color = (text_color_0, text_color_1, text_color_2),
                       caption_height= height,
                       meme_bg_color= (bg_color_0, bg_color_1, bg_color_2))(x)
        return x.resize((256,256))
    
class RandomOverlayEmoji:
    def __init__(self, path = '/raid/VSC/data/emoji/', opacity=[0.2, 1], emoji_size=[0.2, 1], x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.path = path
        self.opacity = opacity
        self.emoji_size = emoji_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __call__(self, x):
        emoji_path = self.path + random.choice(os.listdir(self.path))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        emoji_size = random.uniform(self.emoji_size[0], self.emoji_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = OverlayEmoji(emoji_path = emoji_path,
                         opacity = opacity,
                         emoji_size = emoji_size,
                         x_pos = x_pos,
                         y_pos = y_pos)(x)
        return x
    
class RandomOverlayText(object):
    def __init__(self, text = [0,20], color_1=[0,255], color_2=[0,255], color_3=[0,255], font_size = [0, 1], opacity=[0, 1], x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.text = text
        self.color_1 = color_1
        self.color_2 = color_2
        self.color_3 = color_3
        self.opacity = opacity
        self.font_size = font_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __call__(self, x):
        text = random.choices(range(100), k = random.randint(self.text[0],self.text[1]))
        color = [random.randint(self.color_1[0],self.color_1[1]),
                 random.randint(self.color_2[0],self.color_2[1]),
                 random.randint(self.color_3[0],self.color_3[1])]
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        font_size = random.uniform(self.font_size[0], self.font_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = OverlayText(text = text,
                        font_size = font_size,
                        opacity = opacity,
                        color = color,
                        x_pos = x_pos,
                        y_pos = y_pos)(x)
        return x
    
class RandomPerspectiveTransform:
    def __init__(self, sigmas = [10, 50]):
        self.sigmas = sigmas
    def __call__(self, x):
        sigma = random.uniform(self.sigmas[0], self.sigmas[1])
        x = PerspectiveTransform(sigma=sigma)(x)
        return x
    
class RandomOverlayImage:
    def __init__(self, path = '/raid/VSC/data/training_images_9//', opacity=[0.6, 1], overlay_size=[0.5, 1], x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.path = path
        self.opacity = opacity
        self.overlay_size = overlay_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __call__(self, x):
        bg = Image.open(self.path + random.choice(os.listdir(self.path)))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        overlay_size = random.uniform(self.overlay_size[0], self.overlay_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        bg = OverlayImage(overlay = x,
                         opacity = opacity,
                         overlay_size = overlay_size,
                         x_pos = x_pos,
                         y_pos = y_pos)(bg)
        return bg.resize((256,256))
    
class RandomStackImage:
    def __init__(self, path = '/raid/VSC/data/training_images_9/', choice_1 = [0, 1], choice_2 = [0, 1]):
        self.path = path
        self.choice_1 = choice_1
        self.choice_2 = choice_2

    def __call__(self, x):
        bg = Image.open(self.path + random.choice(os.listdir(self.path)))
        choice_1 = random.randint(self.choice_1[0],self.choice_1[1])
        choice_2 = random.randint(self.choice_2[0],self.choice_2[1])
        
        if choice_1 == 0:
            image1 = x.resize((256,256))
            image2 = bg.resize((256,256))
        else:
            image1 = bg.resize((256,256))
            image2 = x.resize((256,256))
        
        if choice_2 ==0:
            new_image = Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height)))
            new_image.paste(image1, (0, 0))
            new_image.paste(image2, (image1.width, 0))
            new_image = new_image.resize((256,256))
        else:
            new_image = Image.new('RGB', (max(image1.width, image2.width), image1.height + image2.height))
            new_image.paste(image1, (0, 0))
            new_image.paste(image2, (0, image1.height))
            new_image = new_image.resize((256,256))
        return new_image


############ NEW ##############
class RandomPadSquare(object):
    def __init__(self, color_1=[0,255], color_2=[0,255], color_3=[0,255]):
        self.color_1 = color_1
        self.color_2 = color_2
        self.color_3 = color_3

    def __call__(self, x):
        color = tuple([random.randint(self.color_1[0],self.color_1[1]),
                 random.randint(self.color_2[0],self.color_2[1]),
                 random.randint(self.color_3[0],self.color_3[1])])
        
        x = PadSquare(color = color)(x)
        return x.resize((256,256))
    

class RandomChangeChannel(object):
    def __init__(self, shift_v = [0,40], order_v = [0,1,2], invert_0 = [0,1], invert_1 = [0,1], invert_2 = [0,1]):
        self.shift_v = shift_v
        self.order_v = order_v
        self.invert_0 = invert_0
        self.invert_1 = invert_1
        self.invert_2 = invert_2

    def shift_channels(self, image, shift_v=[20, -20]):
        image = np.array(image) 
        if len(image.shape) == 3 and image.shape[-1] == 3:
            H_i,W_i,C_i = image.shape
            I, J = shift_v

            image[:,:,0] = np.roll(image[:,:,0], (I, J), axis=(0,1) )
            image[:,:,2] = np.roll(image[:,:,2], (-I, -J), axis=(0,1) )

            I, J = abs(I), abs(J)
            if I>0 and J>0:
                image = image[I:-I,J:-J] 
            elif I==0 and J>0:
                image = image[:,J:-J] 
            elif I>0 and J==0:
                image = image[I:-I] 
        return Image.fromarray(image)
    
    def swap_channels(self, image, new_channel_order_v=[2,1,0]):
        image = np.array(image) 
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = image[:,:,new_channel_order_v]

        return Image.fromarray(image)
    
    def invert_channel(self, image, invert_r=False, invert_g=False,invert_b=True):
        image = np.array(image) 
        if len(image.shape) == 3 and image.shape[-1] == 3:
            if invert_r:
                image[:,:,0] = 255 - image[:,:,0]

            if invert_g:
                image[:,:,1] = 255 - image[:,:,1]

            if invert_b:
                image[:,:,2] = 255 - image[:,:,2]
        return Image.fromarray(image) 

    def __call__(self, x):
        
        select = random.choices(range(3), k=1)[0]
        if select==0:
            sv = int(random.uniform(self.shift_v[0], self.shift_v[1]))
            x = self.shift_channels(x, shift_v=[sv, -sv])
        if select==1:
            ls = [0,1,2]
            random.shuffle(ls)
            x = self.swap_channels(x, new_channel_order_v=ls)
        if select==2:
            invert_r = random.randint(self.invert_0[0],self.invert_0[1])
            invert_g = random.randint(self.invert_1[0],self.invert_1[1])
            invert_b = random.randint(self.invert_2[0],self.invert_2[1])
            x = self.invert_channel(x, invert_r=invert_r, invert_g=invert_g, invert_b=invert_b)
        
        return x.resize((256,256))

    
class RandomEncodingQuality(object):
    def __init__(self, quality = [0, 50]):
        self.quality = quality
    def __call__(self, x):
        quality = int(random.uniform(self.quality[0], self.quality[1]))
        x = EncodingQuality(quality=quality)(x)
        return x

    
class RandomOverlayStripes(object):
    def __init__(self, line_widths = [0, 1], \
                 line_color_0 = [0, 255], line_color_1 = [0, 255], line_color_2 = [0, 255], \
                 line_angles = [0, 360], line_densitys = [0, 1], line_opacitys = [0, 1]):
        self.line_widths = line_widths
        self.line_color_0 = line_color_0
        self.line_color_1 = line_color_1
        self.line_color_2 = line_color_2
        self.line_angles = line_angles
        self.line_densitys = line_densitys
        self.line_opacitys = line_opacitys

    def __call__(self, x):
        line_width = random.uniform(self.line_widths[0], self.line_widths[1])
        line_color = (random.randint(self.line_color_0[0], self.line_color_0[1]), \
                       random.randint(self.line_color_1[0], self.line_color_1[1]), \
                       random.randint(self.line_color_2[0], self.line_color_2[1]))
        line_angle = random.randint(self.line_angles[0], self.line_angles[1])
        line_density = random.uniform(self.line_densitys[0], self.line_densitys[1])
        line_opacity = random.uniform(self.line_opacitys[0], self.line_opacitys[1])

        x = OverlayStripes(line_width = line_width, \
                           line_color = line_color, \
                           line_angle = line_angle, \
                           line_density = line_density, \
                           line_opacity = line_opacity)(x)

        return x

    
class RandomSharpen(object):
    def __init__(self, factors = [2, 20]):
        self.factors = factors

    def __call__(self, x):
        factor = random.uniform(self.factors[0], self.factors[1])
        x = Sharpen(factor = factor)(x)
        return x

    
class RandomSkew(object):
    def __init__(self, skew_factors = [-2, 2]):
        self.skew_factors = skew_factors

    def __call__(self, x):
        skew_factor = random.uniform(self.skew_factors[0], self.skew_factors[1])
        x = Skew(skew_factor = skew_factor)(x)
        return x

    
class RandomShufflePixels(object):
    def __init__(self, factors = [0.1, 0.5]):
        self.factors = factors

    def __call__(self, x):
        factor = random.uniform(self.factors[0], self.factors[1])
        x = ShufflePixels(factor = factor)(x)
        return x
############ NEW ##############
    
    
    
path_1 = '/raid/VSC/data/training_images/'
path_2 = '/raid/VSC/images/train_v1_s3_all/train_v1_s3_all/'

names = sorted(os.listdir(path_1))
os.makedirs(path_2, exist_ok=True)

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * 4000
end = (num+1) * 4000

for i in range(begin, end):
    if i%10 == 0:
        print('processing...',i)
        image = Image.open(path_1 + names[i])
        name = str(i//10)+'_0.jpg'
        image.resize((256,256)).save(path_2 + name, quality=100)
        for j in range(1,20):
            transform_q = transforms.Compose(
                [ToRGB()] + 
                random.sample([
                    RandomChangeChannel(),
                    RandomPadSquare(),
                    RandomResizeCrop(),
                    RandomRotate(),
                    Random_image_opaque(),
                    GrayScale(),
                    Color_jitter(),
                    RandomBlur(),
                    RandomPixelization(),
                    RandomPad(),
                    RandomAddNoise(),
                    VertFlip(),
                    HoriFlip(),
                    RandomEncodingQuality(),
                    RandomOverlayStripes(),
                    RandomSharpen(),
                    RandomSkew(),
                    RandomShufflePixels(),
                    RandomMemeFormat(),
                    RandomOverlayEmoji(),
                    RandomOverlayText(),
                    RandomPerspectiveTransform(),
                    RandomOverlayImage(),
                    RandomStackImage()
                ], 3) + 
                [transforms.Resize((256,256)), ToRGB()]
            )
            
            image_q = transform_q(image)
            name = str(i//10)+'_'+ str(j) +'.jpg'
            image_q.save(path_2 + name, quality=100)

