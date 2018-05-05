#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'Webカメラから画像を取得する'
#

import cv2
import argparse
import numpy as np

import Tools.imgfunc as IMG
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='使用するWebカメラのID [default: 0]')
    parser.add_argument('-n', '--capture_num', type=int, default=12,
                        help='キャリブレーションボードの撮影枚数 [default: 12]')
    parser.add_argument('--row', type=int, default=5,
                        help='使用するキャリブレーションボードの行 [default: 5]')
    parser.add_argument('--col', type=int, default=8,
                        help='使用するキャリブレーションボードの列 [default: 8]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='画像とキャリブレーションパラメータの保存先 (default: ./result/)')
    parser.add_argument('--img_rate', '-r', type=float, default=1,
                        help='表示する画像サイズの倍率 [default: 1]')
    parser.add_argument('--lower', action='store_true',
                        help='低画質モード（select timeoutが発生する場合に使用）')
    return parser.parse_args()


def capture(cam_id, num, low):
    cap_imgs = []
    cap = cv2.VideoCapture(args.camera_id)
    if low:
        cap.set(3, 200)
        cap.set(4, 200)
        cap.set(5, 5)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', IMG.resize(frame, args.img_rate))

        key = cv2.waitKey(20) & 0xff
        # Display the resulting frame
        if key == 27 or len(cap_imgs) > num:
            print('exit!')
            break
        elif key == ord('c'):
            print('capture!:', len(cap_imgs))
            cap_imgs.append(frame)

    cap.release()
    cv2.destroyAllWindows()
    return cap_imgs


def findCircles(img, board, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    objp = np.zeros((board[0]*board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findCirclesGrid(gray, board, None)
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, board, corners, ret)

    return objp, corners, ret


def main(args):
    cap_imgs = capture(args.camera_id, args.capture_num, args.lower)

    objpts_all = []
    imgpts_all = []
    board = (args.col, args.row)
    for i in cap_imgs:
        objpts, imgpts, ret = findCircles(i, board)
        if ret:
            objpts_all.append(objpts)
            imgpts_all.append(imgpts)

        cv2.imshow('test', i)
        cv2.waitKey(1000)

    if len(objpts_all) < 5:
        print('[Error] not found circles grid board')
        exit()

    img = cap_imgs[0]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpts_all, imgpts_all, gray.shape[::-1], None, None
    )
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )
    print(newcameramtx)
    path = F.getFilePath(args.out_path, 'cam_mat')
    np.savez(path, mat=newcameramtx)
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5
    )
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    path = F.getFilePath(args.out_path, 'result', '.png')
    cv2.imwrite(path, dst)
    cv2.imshow('calibresult', dst)
    cv2.waitKey()
    mean_error = 0
    for i in range(len(objpts_all)):
        imgpts2, _ = cv2.projectPoints(
            objpts_all[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpts_all[i], imgpts2, cv2.NORM_L2)/len(imgpts2)
        mean_error += error

    error = mean_error/len(objpts_all)
    print("total error: {0:8.5f}".format(error))
    param = F.args2dict(args)
    param['error'] = error
    param['mat_fx'] = newcameramtx[0, 0]
    param['mat_fy'] = newcameramtx[1, 1]
    param['mat_cx'] = newcameramtx[0, 2]
    param['mat_cy'] = newcameramtx[1, 2]
    F.dict2json(args.out_path, 'calib_param', param)


if __name__ == '__main__':

    args = command()
    F.argsPrint(args)

    print('Key bindings')
    print('[Esc] Exit')
    print('[ c ] capture')

    main(args)
