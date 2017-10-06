# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from ...proto import FeatureTypes_pb2 as ft

def raise_error_unsupported_categorical_option(option_name, option_value, layer_type, layer_name):
    """
    Raise an error if an option is not supported.
    """
    raise RuntimeError("Unsupported option %s=%s in layer %s(%s)" % (option_name, option_value,
        layer_type, layer_name))

def raise_error_unsupported_option(option, layer_type, layer_name):
    """
    Raise an error if an option is not supported.
    """
    raise RuntimeError("Unsupported option =%s in layer %s(%s)" % (option,
        layer_type, layer_name))

def raise_error_unsupported_scenario(message, layer_type, layer_name):
    """
    Raise an error if an scenario is not supported.
    """
    raise RuntimeError("Unsupported scenario '%s' in layer %s(%s)" % (message, 
        layer_type, layer_name))


def _convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError(
                "{} is not a multiarray type".format(output.name,)
            )
        array_shape = tuple(output.type.multiArrayType.shape)
        if len(array_shape) == 2:
            width, height = array_shape
            output.type.imageType.colorSpace = \
                ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        else:
            channels, width, height = array_shape

            if channels == 1:
                output.type.imageType.colorSpace = \
                    ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
            elif channels == 3:
                if is_bgr:
                    output.type.imageType.colorSpace = \
                        ft.ImageFeatureType.ColorSpace.Value('BGR')
                else:
                    output.type.imageType.colorSpace = \
                        ft.ImageFeatureType.ColorSpace.Value('RGB')
            else:
                raise ValueError(
                    "Channel Value {} not supported for image inputs"
                    .format(channels,)
                )

        output.type.imageType.width = width
        output.type.imageType.height = height
