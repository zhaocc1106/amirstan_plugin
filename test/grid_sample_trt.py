import ctypes
import os
import time
import typing

import numpy as np
import onnx
import pycuda.driver as cuda
import onnxruntime as ort
import tensorrt as trt
import torch
from torch.onnx import symbolic_helper

from src.facerender.animate import AnimateFromCoeff

# called to init pyCUDA
import pycuda.autoinit

_registered_ops: typing.AbstractSet[str] = set()

_OPSET_VERSION = 11


def _reg(symbolic_fn: typing.Callable):
    name = "::%s" % symbolic_fn.__name__
    torch.onnx.register_custom_op_symbolic(name, symbolic_fn, _OPSET_VERSION)
    _registered_ops.add(name)


def register():
    """Register ONNX Runtime's built-in contrib ops.
    Should be run before torch.onnx.export().
    """

    def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
        # mode
        #   'bilinear'      : onnx::Constant[value={0}]
        #   'nearest'       : onnx::Constant[value={1}]
        #   'bicubic'       : onnx::Constant[value={2}]
        # padding_mode
        #   'zeros'         : onnx::Constant[value={0}]
        #   'border'        : onnx::Constant[value={1}]
        #   'reflection'    : onnx::Constant[value={2}]
        mode = symbolic_helper._maybe_get_const(mode, "i")
        padding_mode = symbolic_helper._maybe_get_const(padding_mode, "i")
        mode_str = ["bilinear", "nearest", "bicubic"][mode]
        padding_mode_str = ["zeros", "border", "reflection"][padding_mode]
        align_corners = int(symbolic_helper._maybe_get_const(align_corners, "b"))

        # From opset v13 onward, the output shape can be specified with
        # (N, C, H, W) (N, H_out, W_out, 2) => (N, C, H_out, W_out)
        # input_shape = input.type().sizes()
        # gird_shape = grid.type().sizes()
        # output_shape = input_shape[:2] + gird_shape[1:3]
        # g.op(...).setType(input.type().with_sizes(output_shape))

        return g.op(
            ## op name, modify here. not sure whether "com.microsoft::" is required
            "com.microsoft::GridSamplePluginDynamic",
            input,
            grid,
            # mode_s=mode_str,
            # padding_mode_s=padding_mode_str,
            mode_i=mode,
            padding_mode_i=padding_mode,
            align_corners_i=align_corners,
        )

    _reg(grid_sampler)


class MyGridSample(torch.nn.Module):
    def __init__(self):
        super(MyGridSample, self).__init__()

    def forward(self, inp, grid):
        return torch.nn.functional.grid_sample(inp, grid)


if __name__ == "__main__":
    inp = torch.arange(0, 64)
    inp = inp.reshape((1, 1, 4, 4, 4)).float()
    print('inp.shape: {}'.format(inp.shape))
    out_h = 4
    out_w = 4
    out_d = 4
    new_h = torch.linspace(-1, 1, out_h).view(-1, 1, 1).repeat(1, out_w, out_d)
    new_w = torch.linspace(-1, 1, out_w).view(1, -1, 1).repeat(out_h, 1, out_d)
    new_d = torch.linspace(-1, 1, out_d).view(1, 1, -1).repeat(out_h, out_w, 1)
    grid = torch.cat((new_w.unsqueeze(3), new_h.unsqueeze(3), new_d.unsqueeze(3)), dim=3)
    grid = grid.unsqueeze(0)
    print('grid.shape: {}'.format(grid.shape))

    model = MyGridSample()
    model.eval()
    model = model.cuda()
    # infer
    out = model(inp.cuda(), grid.cuda())
    # print(out.cpu().numpy())

    # 转换成ONNX
    model_onnx_path = "./my_grid_sample.onnx"
    register()  # 注册自定义grid_sampler op
    print('Exporting model to ONNX format...')
    with torch.no_grad():
        torch.onnx.export(model, (inp, grid), model_onnx_path,
                          verbose=True, input_names=['inp', 'grid'], output_names=['out'],
                          opset_version=_OPSET_VERSION,
                          dynamic_axes={'inp': [0, 1, 2, 3, 4], 'grid': [0, 1, 2, 3], 'out': [0, 1, 2, 3, 3]})
    print('Model exported to ' + model_onnx_path)
    model_onnx = onnx.load(model_onnx_path)
    onnx.checker.check_model(model_onnx)
    print('Onnx model graph:')
    print(onnx.helper.printable_graph(model_onnx.graph))
    # onnx推理
    # print('Running inference on ONNX model...')
    # ort_session = ort.InferenceSession(model_onnx_path)
    # ort_inputs = {'inp': inp.cpu().numpy(), 'grid': grid.cpu().numpy()}
    # ort_outs = ort_session.run(None, ort_inputs)

    # 转换成TRT
    cmd = 'trtexec --onnx={} --saveEngine={} ' \
          '--minShapes=inp:1x1x4x4x4,grid:1x4x4x4x3 ' \
          '--maxShapes=inp:1x1x4x4x4,grid:1x4x4x4x3 ' \
          '--optShapes=inp:1x1x4x4x4,grid:1x4x4x4x3 ' \
          '--exportLayerInfo=./generator_trt_layer_info.json ' \
          '--plugins=/usr/local/lib/libamirstan_plugin.so --skipInference' \
        .format('my_grid_sample.onnx', './my_grid_sample.trt')
    os.system(cmd)

    # trt gpu推理
    # 加载引擎
    ctypes.CDLL("/usr/local/lib/libamirstan_plugin.so")
    with open('./my_grid_sample.trt', 'rb') as f:
        trt_engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        inspector = trt_engine.create_engine_inspector()
        print('trt_engine layer_info:\n{}'.format(
            inspector.get_engine_information(trt.LayerInformationFormat(1))
        ))
        trt_ctx = trt_engine.create_execution_context()

    # malloc
    d_inp = cuda.mem_alloc(1 * 1 * 4 * 4 * 4 * 4)
    d_grid = cuda.mem_alloc(1 * 4 * 4 * 4 * 3 * 4)
    d_out = cuda.mem_alloc(1 * 1 * 4 * 4 * 4 * 4)
    h_out = cuda.pagelocked_empty((1, 1, 4, 4, 4), dtype=np.float32)

    # create a stream in which to copy inputs/outputs and run inference
    stream = cuda.Stream()

    # set shape
    idx_inp = trt_engine['inp']
    idx_grid = trt_engine['grid']
    trt_ctx.set_binding_shape(idx_inp, (1, 1, 4, 4, 4))
    trt_ctx.set_binding_shape(idx_grid, (1, 4, 4, 4, 3))

    # 将数据从cpu拷贝到gpu
    inp_ca = np.ascontiguousarray(inp.cpu().numpy())
    grid_ca = np.ascontiguousarray(grid.cpu().numpy())
    cuda.memcpy_htod_async(d_inp, inp_ca, stream)
    cuda.memcpy_htod_async(d_grid, grid_ca, stream)

    # 执行推理
    trt_ctx.execute_async_v2(
        bindings=[int(d_inp), int(d_grid), int(d_out)], stream_handle=stream.handle)

    # 将结果从gpu拷贝到cpu
    out_volume = trt.volume((1, 1, 4, 4, 4))
    cuda.memcpy_dtoh_async(h_out[:out_volume], d_out, stream)

    # stream sync
    stream.synchronize()

    # 验证结果
    print('out: \n{}'.format(out.cpu().numpy()))
    # print('onnx out: {}'.format(ort_outs[0]))
    print('trt_out: \n{}'.format(h_out))
