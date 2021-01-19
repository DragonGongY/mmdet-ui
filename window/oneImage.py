from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def one_image(img, config, checkpoint, score_thr, score_bbox):
    parser = ArgumentParser()

    parser.add_argument('--img', default=img, help='Image file')
    parser.add_argument('--config', default=config, help='Config file')
    parser.add_argument('--checkpoint', default=checkpoint, help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=score_bbox, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=score_thr)
    return result

if __name__ == '__main__':
    one_image()
