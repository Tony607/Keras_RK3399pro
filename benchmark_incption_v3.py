import time
import platform
import numpy as np
import cv2
from rknn.api import RKNN


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = "Inception_v3\n-----TOP 5-----\n"
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = "{}: {}\n".format(index[j], value)
            else:
                topi = "-1: 0.0\n"
            top5_str += topi
    print("top5_str:`{}`".format(top5_str))


if __name__ == "__main__":

    # Create RKNN object
    rknn = RKNN()
    img_height = 299
    # Direct Load RKNN Model
    print("--> Loading RKNN model")
    ret = rknn.load_rknn("./inception_v3.rknn")
    if ret != 0:
        print("Load inception_v3.rknn failed!")
        exit(ret)

    # Set inputs
    img = cv2.imread("./data/beaver.jpg")
    img = cv2.resize(img, dsize=(img_height, img_height), interpolation=cv2.INTER_CUBIC)

    # This can opt out if "reorder_channel" is set to "2 1 0"
    # rknn.config() in `convert_rknn.py`
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print("--> Init runtime environment")

    if "aarch64" in platform.platform():
        target = "rk3399pro"
    else:
        target = None

    ret = rknn.init_runtime(target=target)
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)

    # Inference
    print("--> Running model")
    outputs = rknn.inference(inputs=[img])

    outputs = np.array(outputs)
    print("Outputs shape: {}".format(outputs.shape))
    show_outputs(outputs)

    # Benchmark model
    print("--> Benchmark model")

    times = []

    # Run inference 20 times and do the average.
    for i in range(20):
        start_time = time.time()
        # Use the API internal call directly.
        results = rknn.rknn_base.inference(
            inputs=[img], data_type="uint8", data_format="nhwc", outputs=None
        )
        # Alternatively, use the external API call.
        # outputs = rknn.inference(inputs=[img])
        delta = time.time() - start_time
        times.append(delta)

    # Calculate the average time for inference.
    mean_delta = np.array(times).mean()

    fps = 1 / mean_delta
    print("average(sec):{:.3f},fps:{:.2f}".format(mean_delta, fps))

    # perf
    print("--> Begin evaluate model performance")
    perf_results = rknn.eval_perf(inputs=[img])

    rknn.release()
