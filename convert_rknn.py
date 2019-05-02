import numpy as np
import cv2
from rknn.api import RKNN


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = "inception_v3\n-----TOP 5-----\n"
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
    print("top5_str: {}".format(top5_str))


if __name__ == "__main__":
    INPUT_NODE = ["input_1"]
    OUTPUT_NODE = ["predictions/Softmax"]

    img_height = 299

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print("--> config model")
    # channel_mean_value "0 0 0 255" while normalize the image data to range [0, 1]
    # channel_mean_value "128 128 128 128" while normalize the image data to range [-1, 1]
    # reorder_channel "0 1 2" will keep the color channel, "2 1 0" will swap the R and B channel,
    # i.e. if the input is BGR loaded by cv2.imread, it will convert it to RGB for the model input.
    # need_horizontal_merge is suggested for inception models (v1/v3/v4).
    rknn.config(
        channel_mean_value="128 128 128 128",
        reorder_channel="0 1 2",
        need_horizontal_merge=True,
        quantized_dtype="asymmetric_quantized-u8",
    )

    # Load tensorflow model
    print("--> Loading model")
    ret = rknn.load_tensorflow(
        tf_pb="./model/frozen_model.pb",
        inputs=INPUT_NODE,
        outputs=OUTPUT_NODE,
        input_size_list=[[img_height, img_height, 3]],
    )
    if ret != 0:
        print("Load inception_v3 failed!")
        exit(ret)

    # Build model
    print("--> Building model")
    # dataset: A input data set for rectifying quantization parameters.
    ret = rknn.build(do_quantization=True, dataset="./dataset.txt")
    if ret != 0:
        print("Build inception_v3 failed!")
        exit(ret)

    # Export rknn model
    print("--> Export RKNN model")
    ret = rknn.export_rknn("./inception_v3.rknn")
    if ret != 0:
        print("Export inception_v3.rknn failed!")
        exit(ret)

    # Set inputs
    img = cv2.imread("./data/elephant.jpg")
    img = cv2.resize(img, dsize=(img_height, img_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("--> Init runtime environment")
    ret = rknn.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)

    # Inference
    print("--> Running model")
    outputs = rknn.inference(inputs=[img])
    show_outputs(outputs)
    # print('inference result: ', outputs)

    # perf
    print("--> Begin evaluate model performance")
    perf_results = rknn.eval_perf(inputs=[img])

    rknn.release()
