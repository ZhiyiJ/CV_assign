#!/usr/bin/env python3
import cv2
import numpy as np


def detect_corners(image):
    """Harris corner detector.

  Args:
  - image (2D float64 array): A grayscale image.

  Returns:
  - corners (list of 2-tuples): A list of 2-tuples representing the locations
      of detected corners. Each tuple contains the (x, y) coordinates of a
      pixel, where y is row index and x is the column index, i.e. `image[y, x]`
      gives the corresponding pixel intensity.
  """

    # y is row index
    # x is column index
    # [row,column]=np.shape(image)
    dst = np.zeros((len(image), len(image[0])))
    row = len(image)  # i
    column = len(image[0])  # j
    Ix, Iy = sobel(image)
    Ixx = np.multiply(Ix, Ix)
    Ixy = np.multiply(Ix, Iy)
    Iyy = np.multiply(Iy, Iy)
    for i in range(1, row - 2):
        for j in range(1, column - 2):
            Sxx = np.sum(Ixx[i - 1:i + 2, j - 1:j + 2])
            Syy = np.sum(Iyy[i - 1:i + 2, j - 1:j + 2])
            Sxy = np.sum(Ixy[i - 1:i + 2, j - 1:j + 2])
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            dst[i][j] = det - 0.05 * (trace ** 2)
    dst[dst < 0] = 0  # apply treshold
    dst = nonmaxsup(dst.astype('float64'), 3)
    corners = np.where(dst > 0.06* dst.max())
    corners = [corners[1], corners[0]]
    corners = list(zip(corners[0], corners[1]))

    return corners

    raise NotImplementedError


def sobel(image):
    row = len(image)
    column = len(image[0])
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    Ix = np.zeros((len(image), len(image[0])))
    Iy = np.zeros((len(image), len(image[0])))
    for i in range(1, column - 2):
        for j in range(1, row - 2):
            temp_matrix = [image[j - 1, i - 1:i + 2], image[j, i - 1:i + 2], image[j + 1, i - 1:i + 2]]
            Ix[j][i] = np.sum(np.multiply(sobel_x, temp_matrix))
            Iy[j][i] = np.sum(np.multiply(sobel_y, temp_matrix))

    return Ix, Iy


def nonmaxsup(scores, ksize):
    """Apply non-maximum suppression on a score map.

  Args:
  - scores (2D float64 array): A score map.
  - ksize (int): Kernel size of the maximum filter being used.

  Returns:
  - suppressed (2D float64 array): Suppressed version of `scores` where all
      elements except the local maxima are set to 0.
  """
    suppressed = np.copy(scores)
    filtered = maximum_filter(suppressed, (ksize, ksize))
    maxima = (suppressed == filtered)
    suppressed[np.logical_not(maxima)] = 0
    return suppressed


def match_corners(image1, image2, corners1, corners2):
    """Match corners using mutual marriages.

  Args:
  - image1 (2D float64 array): A grayscale image.
  - image2 (2D float64 array): A grayscale image.
  - corners1 (list of 2-tuples): Corners in image1.
  - corners2 (list of 2-tuples): Corners in image2.

  - corners (list of 2-tuples): A list of 2-tuples representing the locations
      of detected corners.

  Returns:
  - matches (list of 2-tuples): A list of 2-tuples representing the matching
      indices. Each tuple contains two integer indices.
  """
    image1 = np.pad(image1, pad_width=5, mode='constant', constant_values=0)
    image2 = np.pad(image2, pad_width=5, mode='constant', constant_values=0)
    scores1_2 = []
    scores2_1 = []
    pads = 11
    for i in corners1:
        feature_1 = image1[i[1]:i[1] + pads, i[0]:i[0] + pads]  # feature in corners1
        scores1 = [np.sum(np.square(feature_1 - image2[j[1]:j[1] + pads, j[0]:j[0] + pads])) for j in corners2]
        scores1 = np.array(scores1)
        position1 = np.where(scores1 == np.min(scores1))
        scores1_2.append(position1[0])
    for i in corners2:
        feature_2 = image2[i[1]:i[1] + pads, i[0]:i[0] + pads]  # feature in corners2
        scores2 = [np.sum(np.square(feature_2 - image1[j[1]:j[1] + pads, j[0]:j[0] + pads])) for j in corners1]
        scores2 = np.array(scores2)
        position2 = np.where(scores2 == np.min(scores2))
        scores2_1.append(position2[0])
    rows, columns = np.shape(scores1_2)
    matches = []
    for i in range(0, rows - 1):
        if i == scores2_1[int(scores1_2[i])]:
            matches.append([i, int(scores1_2[i])])
    matches=tuple(map(tuple, matches))
    return matches

    raise NotImplementedError


def draw_matches(image1, image2, corners1, corners2, matches,
                 outlier_labels):
    """Draw matched corners between images.

  Args:
  - matches (list of 2-tuples)
  - image1 (3D uint8 array): A color image having shape (H1, W1, 3).
  - image2 (3D uint8 array): A color image having shape (H2, W2, 3).
  - corners1 (list of 2-tuples)
  - corners2 (list of 2-tuples)
  - outlier_labels (list of bool)

  Returns:
  - match_image (3D uint8 array): A color image having shape
      (max(H1, H2), W1 + W2, 3).
  """
    img1 = image1
    img2 = image2
    H1, W1, L1 = np.shape(img1)
    H2, W2, L2 = np.shape(img2)
    match_image = np.zeros((max(H1, H2), W1 + W2, 3))

    match_image[0:H1, 0:W1, 0:3] = img1
    match_image[0:H2, W1:W1 + W2, 0:3] = img2

    match_image = 255 * match_image / np.max(match_image)
    color = (0, 255, 0)
    colorline = (255, 0, 0)
    for i in matches:
        center_coordinates1 = (corners1[i[0]][0], corners1[i[0]][1])
        center_coordinates2 = (corners2[i[1]][0] + W1, corners2[i[1]][1])
        match_image = cv2.circle(match_image, center_coordinates1, 5, color, -1)
        match_image = cv2.circle(match_image, center_coordinates2, 5, color, -1)
        match_image = cv2.line(match_image, center_coordinates1, center_coordinates2, colorline, 2)

    if outlier_labels is not None:
        for j in range(0, len(outlier_labels) - 1):
            if outlier_labels[j] == True:
                center_coordinates1 = (corners1[matches[j][0]][0], corners1[matches[j][0]][1])
                center_coordinates2 = (corners2[matches[j][1]][0] + W1, corners2[matches[j][1]][1])
                match_image = cv2.line(match_image, center_coordinates1, center_coordinates2, (0, 0, 255), 2)

    match_image=np.uint8(match_image)

    return match_image

    raise NotImplementedError


def compute_affine_xform(corners1, corners2, matches):
    """Compute affine transformation given matched feature locations.

  Args:
  - corners1 (list of 2-tuples)
  - corners1 (list of 2-tuples)
  - matches (list of 2-tuples)
   - corners (list of 2-tuples): A list of 2-tuples representing the locations
      of detected corners.

  Returns:
  - xform (2D float64 array): A 3x3 matrix representing the affine
      transformation that maps coordinates in image1 to the corresponding
      coordinates in image2.
  - outlier_labels (list of bool): A list of Boolean values indicating whether
      the corresponding match in `matches` is an outlier or not.
  """
    corners1 = np.array(corners1)
    corners2 = np.array(corners2)
    matches = np.array(matches)
    x = corners1[matches[:, 0], 0]
    y = corners1[matches[:, 0], 1]
    xp = corners2[matches[:, 1], 0]
    yp = corners2[matches[:, 1], 1]
    h, w = np.shape(matches)
    A_t = np.zeros((2 * h, 6))
    B_t = np.zeros((2 * h,))
    A_t[0::2, 0] = x
    A_t[0::2, 1] = y
    A_t[0::2, 2] = 1
    A_t[1::2, 3] = x
    A_t[1::2, 4] = y
    A_t[1::2, 5] = 1
    B_t[0::2] = xp
    B_t[1::2] = yp
    max = 0
    inliers = []
    xform = np.zeros((3, 3))
    lables=np.ones((h,), dtype=bool)

    for i in range(0, 30):
        inde=np.random.choice(len(matches),3,replace=False)
        [x1, x2, x3] = x[inde]
        [y1, y2, y3] = y[inde]
        [xp1, xp2, xp3] = xp[inde]
        [yp1, yp2, yp3] = yp[inde]
        A = [[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1], [x2, y2, 1, 0, 0, 0], [0, 0, 0, x2, y2, 1],
             [x3, y3, 1, 0, 0, 0], [0, 0, 0, x3, y3, 1]]
        A = np.array(A)
        B = np.array([xp1, yp1, xp2, yp2, xp3, yp3])
        [a,b,c,d,e,f]=np.dot(np.linalg.inv(A),B)
        coeff = np.dot(np.linalg.inv(A), B)
        difference = np.array((np.dot(A_t, coeff) - B_t))
        difference=np.sqrt(np.square(difference[0::2])+np.square(difference[1::2]))
        inline = np.where(difference < 3)
        inliers = np.append(inliers, inline)
        inliers = np.unique(inliers)

        if max<len(inline[0]):
            max = len(inline[0])
            # find form for that good performance
            xform=np.array([[a,b,c],[d,e,f],[0,0,1]])
    inliers = np.array(np.unique(inliers)) #position in match
    inliers=inliers.astype(int)
    lables[inliers]=False
    outlier_labels=lables
    return outlier_labels, xform

    raise NotImplementedError


# Extra Credit
def compute_proj_xform(corners1, corners2, matches):
    """Compute projective transformation given matched feature locations.

  Args:
  - corners1 (list of 2-tuples)
  - corners1 (list of 2-tuples)
  - matches (list of 2-tuples)

  Returns:
  - xform (2D float64 array): A 3x3 matrix representing the projective
      transformation that maps coordinates in image1 to the corresponding
      coordinates in image2.
  - outlier_labels (list of bool)
  """
    raise NotImplementedError

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def stitch_images(image1, image2, xform):
    """Stitch two matched images given the transformation between them.

  Args:
  - image1 (3D uint8 array): A color image.
  - image2 (3D uint8 array): A color image.
  - xform (2D float64 array): A 3x3 matrix representing the transformation
      between image1 and image2. This transformation should map coordinates
      in image1 to the corresponding coordinates in image2.

  Returns:
  - image_stitched (3D uint8 array)
  """

    rows, cols, ch = image1.shape
    img1 = cv2.warpPerspective(image1, xform, (cols, rows))
    img2 = image2
    H1, W1, L1 = np.shape(img1)
    H2, W2, L2 = np.shape(img2)
    match_image = np.zeros((max(H1, H2), W1 + W2, 3))

    match_image[0:H1, 0:W1, 0:3] = img1
    match_image[0:H2, W1:W1 + W2, 0:3] = img2
    match_image=np.uint8(match_image)
    h, w,l = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.warpPerspective(image1, xform, (image1.shape[1] + image2.shape[1]+50, 50+max(image1.shape[0],image2.shape[0])))
    dst[0:image1.shape[0], 0:image1.shape[1]] = image2
    dst=trim(dst)

    return dst
    raise NotImplementedError





def baseline_main():
    img_path1 = 'data/bikes1.png'
    print(img_path1)
    img_path2 = 'data/bikes3.png'
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    image1=cv2.imread(img_path1, cv2.IMREAD_COLOR)
    image2=cv2.imread(img_path2, cv2.IMREAD_COLOR)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0

    # TODO
    ''''
    Feature detection   
    '''''
    corners1 = detect_corners(gray1)
    corners2 = detect_corners(gray2)
    img1[[item[1] for item in corners1], [item[0] for item in corners1]] = [0, 0, 255]
    img2[[item[1] for item in corners2], [item[0] for item in corners2]] = [0, 0, 255]
    ''''
    Feature matching  
    '''''
    matches = match_corners(gray1, gray2, corners1, corners2)
    print(matches)
    ''''
          Image alignment 
    '''''
    outlier_labels, xform = compute_affine_xform(corners1, corners2, matches)
    print(np.shape(outlier_labels))
    match_image = draw_matches(img1, img2, corners1, corners2, matches,
                               outlier_labels)
    print(xform)
    stitch=stitch_images(image1,image2,xform)
    cv2.imwrite('output/' + 'stitch_image_bike13.png', stitch)
    cv2.imwrite('output/'+'bike13.png',match_image)



if __name__ == '__main__':
    baseline_main()
