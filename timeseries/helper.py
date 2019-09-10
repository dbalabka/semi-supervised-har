from io import BytesIO
import matplotlib.image as mpimg
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt

class NoFitMixin:
    """
    Some estimator do only tranformation. This mixin just mock fit method.
    """

    def fit(self, X, y=None):
        return self


def plot_model_recursive(model, title=None, exclude_models_by_name=None):
    # model.summary()

    if type(exclude_models_by_name) is list:
        if model.name in exclude_models_by_name:
            return
    else:
        exclude_models_by_name = []
    exclude_models_by_name.append(model.name)

    if title is None:
        title = 'Model %s' % model.name

    # render pydot by calling dot, no file saved to disk
    png_str = model_to_dot(model, show_shapes=True).create_png(prog='dot')

    # treat the dot output string as an image file
    sio = BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)

    # set actual size of image on plot
    dpi = 80
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)

    # plot the image
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show(block=False)

    for layer in model.layers:
        if isinstance(layer, Model):
            plot_model_recursive(layer, exclude_models_by_name=exclude_models_by_name)