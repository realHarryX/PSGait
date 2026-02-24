import os
import numpy as np
import cv2
import logging
from multiprocessing import Pool
from tqdm import tqdm
import argparse


# Define body parts
body_parts = [
    {'name': 'nose', 'type': 'circle', 'keypoints': [0], 'color': (0, 0, 255)},
    {'name': 'left_ear', 'type': 'circle', 'keypoints': [3], 'color': (255, 0, 0)},
    {'name': 'right_ear', 'type': 'circle', 'keypoints': [4], 'color': (0, 255, 0)},
    {'name': 'left_upper_arm', 'type': 'line', 'keypoints': [5, 7], 'color': (0, 255, 255)},
    {'name': 'right_upper_arm', 'type': 'line', 'keypoints': [6, 8], 'color': (255, 0, 255)},
    {'name': 'left_lower_arm', 'type': 'line', 'keypoints': [7, 9], 'color': (255, 255, 0)},
    {'name': 'right_lower_arm', 'type': 'line', 'keypoints': [8, 10], 'color': (255, 165, 0)},
    {'name': 'left_thigh', 'type': 'line', 'keypoints': [11, 13], 'color': (128, 0, 128)},
    {'name': 'right_thigh', 'type': 'line', 'keypoints': [12, 14], 'color': (0, 128, 128)},
    {'name': 'left_calf', 'type': 'line', 'keypoints': [13, 15], 'color': (255, 20, 147)},
    {'name': 'right_calf', 'type': 'line', 'keypoints': [14, 16], 'color': (139, 69, 19)},
]


def process_single_file(silhouette_path, keypoint_path, output_path,
                        conf_threshold, circle_r, line_width):

    try:
        sil_img = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)
        if sil_img is None:
            logging.info(f"Failed to read silhouette image: {silhouette_path}")
            return

        height, width = sil_img.shape
        rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_img[sil_img != 0] = [255, 255, 255]

        with open(keypoint_path, 'r') as f:
            keypoints_data = f.read().strip().split(',')

        keypoints = []
        keypoint_scores = []

        for i in range(2, len(keypoints_data), 3):
            x = float(keypoints_data[i])
            y = float(keypoints_data[i + 1])
            score = float(keypoints_data[i + 2])
            keypoints.append([x, y])
            keypoint_scores.append(score)

        keypoints = np.array(keypoints)
        keypoint_scores = np.array(keypoint_scores)

        for part in body_parts:

            if part['type'] == 'circle':
                idx = part['keypoints'][0]
                x, y = keypoints[idx]
                conf = keypoint_scores[idx]

                if conf < conf_threshold:
                    continue

                if not (0 <= x < width and 0 <= y < height):
                    continue

                cv2.circle(rgb_img, (int(x), int(y)),
                           circle_r, part['color'], -1)

            elif part['type'] == 'line':
                idx1, idx2 = part['keypoints']
                x1, y1 = keypoints[idx1]
                x2, y2 = keypoints[idx2]
                conf1 = keypoint_scores[idx1]
                conf2 = keypoint_scores[idx2]

                if conf1 < conf_threshold or conf2 < conf_threshold:
                    continue

                if not (0 <= x1 < width and 0 <= y1 < height):
                    continue
                if not (0 <= x2 < width and 0 <= y2 < height):
                    continue

                cv2.line(rgb_img,
                         (int(x1), int(y1)),
                         (int(x2), int(y2)),
                         part['color'],
                         line_width)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, rgb_img)

    except Exception as e:
        logging.info(f"Error processing {silhouette_path}: {e}")


def find_all_files(root_dir, extension):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths


def main(args):

    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    silhouette_files = find_all_files(args.silhouette_root, '.png')
    keypoint_files = find_all_files(args.keypoint_root, '.txt')

    if len(silhouette_files) != len(keypoint_files):
        logging.info("Mismatch between silhouette and keypoint file numbers.")
        return

    tasks = [
        (
            sil_path,
            kp_path,
            os.path.join(
                args.output_root,
                os.path.relpath(sil_path, args.silhouette_root)
            ),
            args.conf_threshold,
            args.circle_r,
            args.line_width
        )
        for sil_path, kp_path in zip(
            sorted(silhouette_files),
            sorted(keypoint_files)
        )
    ]

    with Pool(processes=args.num_processes) as pool:
        list(tqdm(pool.starmap(process_single_file, tasks),
                  total=len(tasks)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate parsing skeleton images'
    )

    parser.add_argument('--silhouette_root', required=True)
    parser.add_argument('--keypoint_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--log_file', required=True)

    parser.add_argument('--conf_threshold', type=float)
    parser.add_argument('--circle_r', type=int)
    parser.add_argument('--line_width', type=int)

    parser.add_argument('--num_processes', type=int, default=32)

    args = parser.parse_args()
    main(args)