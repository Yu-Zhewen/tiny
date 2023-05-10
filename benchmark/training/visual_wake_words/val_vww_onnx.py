import os
import onnx
import onnxruntime
import pickle
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')

QUANT_MODEL = False
LOG_SPARSITY = True

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
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.05,
      height_shift_range=0.05,
      zoom_range=.1,
      horizontal_flip=True,
      validation_split=0.1,
      rescale=1. / 255)

    val_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(96, 96),
      batch_size=1,
      subset='validation',
      color_mode='rgb')

    _name = "vww_96"
    if QUANT_MODEL:
        onnx_file_name = "trained_models/" + _name + "_int8.onnx"
    else:
        onnx_file_name = "trained_models/" + _name + "_float.onnx"

    model = onnx.load(onnx_file_name)
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
        for v in list(model.graph.value_info) + list(model.graph.input):
            if v.name in n.input and v.type.tensor_type.shape.dim[0].dim_value == 1:
                return v.name
    conv_inputs = {}
    for node in model.graph.node:
        if node.op_type == 'Conv':
            i = _find_act_input(model, node)
            conv_inputs[node.name] = i

    output_data = []
    labels = []

    if LOG_SPARSITY:
        sparsity_logger = SparsityLogger(conv_inputs)
    assert len(pred_names) == 1
    pred_name = pred_names[0]
    pred_idx = output_names.index(pred_name)

    for i, (dat, label) in enumerate(val_generator):
        if i >= val_generator.n:
            break
        if QUANT_MODEL:
            input_scale = 0.003921568859368563
            input_zero_point = -128
            dat_q = np.array(dat/input_scale + input_zero_point, dtype=np.int8)
            input_data = dat_q
        else:
            input_data = dat
        input_data = input_data.transpose((0, 3, 1, 2))

        assert len(input_names) == 1
        inputs = {input_names[0]: input_data}
        outputs = sess.run(output_names, inputs)
        if LOG_SPARSITY:
            sparsity_logger.update(output_names, outputs)
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data.append(np.argmax(outputs[pred_idx]))
        labels.append(label[0][1])

    if LOG_SPARSITY:
        if QUANT_MODEL:
            pickle.dump(sparsity_logger.ma_meter, open(f'trained_models/{_name}_quant_sparsity.pkl', 'wb'))
        else:
            pickle.dump(sparsity_logger.ma_meter, open(f'trained_models/{_name}_sparsity.pkl', 'wb'))

    num_correct = np.sum(np.array(labels) == output_data)
    acc = num_correct / len(labels)

    print(f"Accuracy = {acc:5.3f} ({num_correct}/{len(labels)})")