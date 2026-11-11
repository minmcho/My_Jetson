"""ONNX export, TensorRT engine building (FP16 / INT8) and a small TensorRT runner."""
import inspect
from pathlib import Path

import torch

try:
    import tensorrt as trt
    LOGGER = trt.Logger(trt.Logger.WARNING)
    TRT_MAJOR = int(trt.__version__.split(".")[0])
except ImportError:  # e.g. running on a laptop without TensorRT
    trt = None


def export_onnx(model, size, path, opset=17):
    """Export a [0,1]-input backbone with a dynamic batch dimension.
    Models over 2 GB (e.g. Virchow FP32) are written with external weight files."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # classic exporter: stable dynamic_axes behaviour
    dummy = torch.rand(1, 3, size, size)
    with torch.no_grad():
        torch.onnx.export(model.float().cpu().eval(), dummy, str(path),
                          input_names=["input"], output_names=["embedding"],
                          dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=opset, do_constant_folding=True, **kwargs)
    return path


if trt is not None:
    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """Feeds batches of real training tiles so INT8 scales match the data."""

        def __init__(self, batches, cache_path):
            super().__init__()
            self.batches, self.i, self.cache = batches, 0, Path(cache_path)
            self._cur = None

        def get_batch_size(self):
            return int(self.batches[0].shape[0])

        def get_batch(self, names):
            if self.i >= len(self.batches):
                return None
            self._cur = self.batches[self.i].contiguous()
            self.i += 1
            return [int(self._cur.data_ptr())]

        def read_calibration_cache(self):
            return self.cache.read_bytes() if self.cache.exists() else None

        def write_calibration_cache(self, cache):
            self.cache.write_bytes(bytes(cache))


def build_engine(onnx_path, engine_path, precision, size, opt_batch, max_batch,
                 calib_batches=None, workspace_gb=2):
    """Build and save a TensorRT engine. precision: 'fp16' or 'int8' (INT8 + FP16 fallback)."""
    if trt is None:
        raise RuntimeError("TensorRT is not installed")
    engine_path = Path(engine_path)
    if engine_path.exists():
        return engine_path
    builder = trt.Builder(LOGGER)
    flags = 0 if TRT_MAJOR >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, size, size), (opt_batch, 3, size, size),
                      (max_batch, 3, size, size))
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        if not calib_batches:
            raise ValueError("INT8 needs calibration batches")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _EntropyCalibrator(
            calib_batches, engine_path.with_suffix(".calib"))
        config.set_calibration_profile(profile)
    print(f"[trt] building {engine_path.name} (this can take several minutes)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {engine_path.name}")
    engine_path.write_bytes(bytes(serialized))
    return engine_path


class TRTRunner:
    """Run a TensorRT engine on CUDA torch tensors (TensorRT 10 tensor API)."""

    def __init__(self, engine_path):
        if trt is None:
            raise RuntimeError("TensorRT is not installed")
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.out_dtype = torch.float16 if self.engine.get_tensor_dtype(self.out) == trt.float16 \
            else torch.float32

    def __call__(self, x):
        x = x.float().contiguous()
        self.ctx.set_input_shape(self.inp, tuple(x.shape))
        y = torch.empty(tuple(self.ctx.get_tensor_shape(self.out)),
                        dtype=self.out_dtype, device=x.device)
        self.ctx.set_tensor_address(self.inp, int(x.data_ptr()))
        self.ctx.set_tensor_address(self.out, int(y.data_ptr()))
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return y.float()
