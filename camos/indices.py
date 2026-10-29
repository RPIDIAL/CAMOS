"""Landmark index definitions (MediaPipe FaceLandmarker, 478 points).

All lists are taken verbatim from the training code so that the 282-point ordering
fed to the model is identical to the one used for training.
"""

Upper_midline = [151, 9, 8, 168, 6]
Upper_left = [337, 336, 285, 417, 351, 412, 465, 464, 357, 413, 453, 452, 451, 450, 449, 448, 343, 350, 277, 329, 349, 348, 347, 346, 345, 441, 442, 443, 444, 445, 342, 446, 261, 295, 282, 283, 276, 353, 265, 340, 296, 334, 293, 300, 383, 372, 299, 333, 298, 301, 368, 264, 447, 251, 389, 356, 454]
Upper_right = [108, 107, 55, 193, 122, 188, 245, 244, 128, 189, 233, 232, 231, 230, 229, 228, 114, 121, 47, 100, 120, 119, 118, 117, 116, 221, 222, 223, 224, 225, 113, 226, 31, 65, 52, 53, 46, 124, 35, 111, 66, 105, 63, 70, 156, 143, 69, 104, 68, 71, 139, 34, 227, 21, 162, 127, 234]
Lower_midline = [197, 195, 5, 4, 2, 164, 0, 17, 18, 200, 199, 175, 152, 12, 15]
Lower_left = [323, 361, 288, 397, 365, 379, 378, 400, 377, 366, 401, 435, 367, 364, 394, 395, 369, 396, 352, 376, 433, 416, 434, 430, 431, 262, 428, 280, 411, 330, 371, 266, 425, 427, 358, 423, 426, 436, 432, 422, 424, 418, 421, 287, 273, 335, 406, 313, 405, 327, 326, 393, 391, 322, 410, 269, 419, 248, 281, 275, 399, 456, 363, 440, 437, 420, 360, 344, 355, 429, 291, 271, 403, 279]
Lower_right = [93, 132, 58, 172, 136, 150, 149, 176, 148, 137, 177, 215, 138, 135, 169, 170, 140, 171, 123, 147, 213, 192, 214, 210, 211, 32, 208, 50, 187, 101, 142, 36, 205, 207, 129, 203, 206, 216, 212, 202, 204, 194, 201, 57, 43, 106, 182, 83, 181, 98, 97, 167, 165, 92, 186, 39, 196, 3, 51, 45, 174, 236, 134, 220, 217, 198, 131, 115, 126, 209, 61, 41, 179, 49]

upper_idx = Upper_midline + Upper_left + Upper_right   # 119 points
lower_idx = Lower_midline + Lower_left + Lower_right   # 163 points
all_idx = upper_idx + lower_idx                          # 282 points, model input order

# Lip landmarks: only trusted from views whose pitch (rot1) is <= ROT_TH (see landmarks.py).
lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306, 62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324, 96, 89, 179, 86, 15, 316, 403, 319, 325, 77, 90, 180, 85, 16, 315, 404, 320, 307, 146, 91, 181, 84, 17, 314, 405, 321, 375]

# MediaPipe ids used to center the point cloud before normalization (glabella, pronasale, right/left eye).
GB_ID, PRN_ID, RIGHT_EYE_ID, LEFT_EYE_ID = 9, 4, 118, 347

# Positions (in the 282 ordering) excluded from the TPS control points (inner-lip points).
TPS_EXCLUDE_POS = [132, 133, 279, 280, 205, 206]

# MediaPipe ids used for canonical alignment and face cropping.
ALIGN_IDS = [151, 6, 117, 346]
CROP_GB_ID, CROP_GOR_ID, CROP_GOL_ID, CROP_CER_ID = 9, 58, 288, 152
