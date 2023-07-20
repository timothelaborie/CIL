import os
import numpy as np
import matplotlib.image as mpimg
import re
import PIL


def validate_fetches(img_fetches, num_models):
    valid = True
    for img_num in img_fetches.keys():
        if len(img_fetches[img_num]) != num_models:
            valid = False
            break

    return valid


def fetch_from_paths(*args):
    img_fetches = {}
    for dir in args:
        for filename in os.listdir(dir):
            image_filename = os.path.join(dir, filename)
            if os.path.isfile(image_filename):
                img_number = int(re.search(r"\d+", image_filename.replace("-2023", "")).group(0))
                im = PIL.Image.open(image_filename)
                im_arr = np.asarray(im)
                if len(im_arr.shape) > 2:
                    # Convert to grayscale.
                    im = im.convert("L")
                    im_arr = np.asarray(im)

                if img_number not in img_fetches.keys():
                    img_fetches[img_number] = [im_arr]
                else:
                    img_fetches[img_number].append(im_arr)

    if validate_fetches(img_fetches, len(args)):
        return img_fetches
    else:
        print("Inconsistent prediction files across participating models!")

    return None


def ensemble_each_fetch(img_fetches, ensemble_routine):
    img_ensembles = {}
    for img_num in img_fetches.keys():
        img_ensembles[img_num] = ensemble_routine(img_fetches[img_num])

    return img_ensembles

def dump_ensembles(img_ensembles, loc, prefix="satimage_"):
    try:
        os.mkdir(loc)
    except:
        print(loc + " exists. Reusing it!")

    for img_num in img_ensembles.keys():
        PIL.Image.fromarray(img_ensembles[img_num]).convert("L").save(loc + "/" + prefix + str(img_num) + ".png", "PNG")

    return

"""
Write your own custom ensembler for several preds to generate a single ensemble pred
"""

def inverse_sigmoid_ensembler(preds):
    ensemble_pred = np.zeros_like(preds[0])

    def inverse_sigmoid(prob):
        return np.log(prob/(1-prob))

    def sigmoid(logit):
        return 1/(1+np.exp(-logit))

    float_preds = [pred/255.0 for pred in preds]

    h, w = preds[0].shape

    for i in range(h):
        for j in range(w):
            ensemble_pred[i, j] = int(255 * sigmoid(np.mean([inverse_sigmoid(float_pred[i, j]) for float_pred in float_preds])))

    return ensemble_pred


if __name__ == "__main__":
    img_fetches = fetch_from_paths("data/ethz-cil-road-segmentation-2023/test/pred_convnext",
                           "data/ethz-cil-road-segmentation-2023/test/pred_segformer",
                           "data/ethz-cil-road-segmentation-2023/test/pred_swin")
    # print(img_fetches[144][0].shape)
    # dump_ensembles({476: np.zeros((400, 400)), 321: 255*np.ones((400, 400))}, "./test_loc/")
    img_ensembles = ensemble_each_fetch(img_fetches, ensemble_routine=inverse_sigmoid_ensembler)
    dump_ensembles(img_ensembles, loc="data/ethz-cil-road-segmentation-2023/test/pred_inv_sig_ensembles/")
