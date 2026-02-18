import onnxruntime as ort

def onnxLoader(model_path: str, runtime_cfg, logger=None):
    so = ort.SessionOptions()

    try:
        threads = getattr(runtime_cfg, "threads", None)
        if threads is not None:
            intra = getattr(threads, "intra_op", None)
            inter = getattr(threads, "inter_op", None)

            if isinstance(intra, int) and intra > 0:
                so.intra_op_num_threads = intra
            if isinstance(inter, int) and inter > 0:
                so.inter_op_num_threads = inter

        available = ort.get_available_providers()

        onnx_cfg = getattr(runtime_cfg, "onnx", None)
        cfg_providers = getattr(onnx_cfg, "providers", None) if onnx_cfg else None

        device = (getattr(runtime_cfg, "device", "cpu") or "cpu").lower()

        if cfg_providers:
            desired = list(cfg_providers)
        else:
            if device == "gpu":
                desired = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                desired = ["CPUExecutionProvider"]

        providers = [p for p in desired if p in available]

        if not providers:
            providers = ["CPUExecutionProvider"]

        provider_options = None
        cfg_opts = getattr(onnx_cfg, "provider_options", None) if onnx_cfg else None
        if isinstance(cfg_opts, dict) and cfg_opts:
            provider_options = [cfg_opts.get(p, {}) for p in providers]

        if provider_options is not None:
            session = ort.InferenceSession(model_path, sess_options=so,
                                        providers=providers, provider_options=provider_options)
        else:
            session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        logger.info("ORT available providers:", available)
        logger.info("ORT desired providers:", desired)
        logger.info("ORT session providers:", session.get_providers())

        input_name = session.get_inputs()[0].name

    except Exception as e:
        logger.exception(e)

    def predict_fn(x):
        # devuelve numpy arrays - NO tolist
        return session.run(None, {input_name: x})

    return predict_fn
