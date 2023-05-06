'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

onnx_test.py: converted models performances on cifar10 test set
'''

import tensorflow as tf
import numpy as np
import h5py
import os
import sys
import train
import eval_functions_eembc
from sklearn.metrics import roc_auc_score
import keras_model
import onnx
import onnxruntime
import pickle

np.set_printoptions(threshold=sys.maxsize)

# if True uses the official MLPerf Tiny subset of CIFAR10 for validation
# if False uses the full CIFAR10 validation set
PERF_SAMPLE = True
# if True uses quantized model
QUANT_MODEL = True
# if True gathers activation sparsity
LOG_SPARSITY = True

if QUANT_MODEL:
    _name = keras_model.get_quant_model_name()
    model_path = 'trained_models/' + _name + '_quant.onnx'
else:
    _name = keras_model.get_quant_model_name()
    model_path = 'trained_models/' + _name + '.onnx'

class SparsityLogger:
    def __init__(self, conv_inputs):
        self.conv_inputs = conv_inputs
        self.ma_meter = {k : 0 for k in conv_inputs.keys()}
        self.samp_counter = 0

    def update(self, output_names, outputs):
        for conv_name, act_name in self.conv_inputs.items():
            output_idx = output_names.index(act_name)
            act = outputs[output_idx]
            sparsity = (act.size - np.count_nonzero(act)) / act.size
            self.ma_meter[conv_name] = (self.ma_meter[conv_name] * self.samp_counter + sparsity) / (self.samp_counter + 1)
        self.samp_counter += 1

if __name__ == '__main__':
    # Load the ONNX model.
    model = onnx.load(model_path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1 # batch size
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
    model.graph.ClearField('value_info')
    model = onnx.shape_inference.infer_shapes(model)
    pred_names = [i.name for i in model.graph.output]

    if LOG_SPARSITY:
        # Add intermediate layer outputs.
        for node in model.graph.node:
            layer_info = onnx.helper.ValueInfoProto()
            layer_info.name = node.output[0]
            model.graph.output.append(layer_info)
        layer_info = onnx.helper.ValueInfoProto()
        layer_info.name = model.graph.input[0].name
        model.graph.output.append(layer_info)
    sess = onnxruntime.InferenceSession(model.SerializeToString())

    # Get input and output tensors.
    input_names = [i.name for i in model.graph.input]
    output_names = [i.name for i in model.graph.output]

    # Get activation tensors
    def _find_act_input(model, n):
        for v in model.graph.value_info:
            if v.name in n.input and v.type.tensor_type.shape.dim[0].dim_value == 1:
                    return v.name
    conv_inputs = {}
    for node in model.graph.node:
        if node.op_type == 'Conv':
            i = _find_act_input(model, node)
            conv_inputs[node.name] = i

    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_imgs, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    if PERF_SAMPLE:
        _idxs = np.load('perf_samples_idxs.npy')
        test_imgs = test_imgs[_idxs]
        test_labels = test_labels[_idxs]
        test_filenames = test_filenames[_idxs]

    label_classes = np.argmax(test_labels, axis=1)
    print("Label classes: ", label_classes.shape)

    if QUANT_MODEL:
        test_imgs = test_imgs.astype(np.int64) - 128
        test_imgs = test_imgs.astype(np.int8)
    else:
        test_imgs = test_imgs.astype(np.float32)

    if LOG_SPARSITY:
        sparsity_logger = SparsityLogger(conv_inputs)

    assert len(pred_names) == 1
    pred_name = pred_names[0]
    pred_idx = output_names.index(pred_name)
    predictions = []
    for img in test_imgs:
        input_data = img.reshape(1, 32, 32, 3)
        input_data = input_data.transpose((0, 3, 1, 2))
        assert len(input_names) == 1
        inputs = {input_names[0]: input_data}
        outputs = sess.run(output_names, inputs)
        if LOG_SPARSITY:
            sparsity_logger.update(output_names, outputs)
        predictions.append(outputs[pred_idx].reshape(10,))
    predictions = np.array(predictions)

    if LOG_SPARSITY:
        pickle.dump(sparsity_logger.ma_meter, open(f'trained_models/{_name}_sparsity.pkl', 'wb'))

    print("EEMBC calculate_accuracy method")
    accuracy_eembc = eval_functions_eembc.calculate_accuracy(predictions, label_classes)
    print("---------------------")

    auc_scikit = roc_auc_score(test_labels, predictions)
    print("sklearn.metrics.roc_auc_score method")
    print("AUC sklearn: ", auc_scikit)
    print("---------------------")

    print("EEMBC calculate_auc method")
    auc_eembc = eval_functions_eembc.calculate_auc(predictions, label_classes, label_names, model_path)
    print("---------------------")
