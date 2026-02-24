def visualize_mask_difference(mask_orig, mask_perturb, save_path="mask_diff.png"):
    """
    So sánh 2 mask nhị phân và tô màu điểm khác nhau:
    - Xanh lá: chỉ có ở mask gốc
    - Đỏ: chỉ có ở mask perturb
    - Trắng: có ở cả hai
    """

    # Ensure uint8
    mask_orig = mask_orig.astype(np.uint8)
    mask_perturb = mask_perturb.astype(np.uint8)

    # Nhị phân hoá (phòng trường hợp mask không đúng 0/255)
    _, mask_orig = cv2.threshold(mask_orig, 127, 255, cv2.THRESH_BINARY)
    _, mask_perturb = cv2.threshold(mask_perturb, 127, 255, cv2.THRESH_BINARY)

    h, w = mask_orig.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # White pixels
    orig_white = mask_orig == 255
    perturb_white = mask_perturb == 255

    # Chỉ có ở mask gốc → xanh lá
    output[np.logical_and(orig_white, ~perturb_white)] = (0, 255, 0)

    # Chỉ có ở mask perturb → đỏ
    output[np.logical_and(~orig_white, perturb_white)] = (0, 0, 255)

    # Có ở cả hai → trắng
    output[np.logical_and(orig_white, perturb_white)] = (255, 255, 255)

    cv2.imwrite(save_path, output)
    return output