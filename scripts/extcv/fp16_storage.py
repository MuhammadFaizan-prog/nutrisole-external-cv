"""Store large FP32 initializers as FP16 + Cast(to=FLOAT) so the file halves in size while compute stays FP32
(ORT constant-folds the Casts at session init). Inputs/outputs stay float32. The export "date" metadata is removed
so the output is reproducible."""
import sys, onnx, numpy as np
from onnx import numpy_helper, helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
new_inits, casts = [], []
for init in g.initializer:
    if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= 16:
        arr = numpy_helper.to_array(init)
        h = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
        new_inits.append(h)
        casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
    else:
        new_inits.append(init)
del g.initializer[:]; g.initializer.extend(new_inits)
# Ultralytics stamps the export time into metadata; drop it so the same weights always give the same file hash.
keep = [p for p in m.metadata_props if p.key != "date"]; del m.metadata_props[:]; m.metadata_props.extend(keep)
nodes = list(g.node); del g.node[:]; g.node.extend(casts + nodes)
onnx.checker.check_model(m); onnx.save(m, dst); print(dst, len(casts), "initializers stored as fp16")
