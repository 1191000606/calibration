import pyrealsense2 as rs
import numpy as np
import cv2


class Camera:
    def __init__(self,
                 color_res,
                 depth_res,
                 frame_rate):
        self.color_res = color_res
        self.depth_res = depth_res
        self.frame_rate = frame_rate

        # build camera pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, *self.color_res, rs.format.rgb8, self.frame_rate)
        config.enable_stream(rs.stream.depth, *self.depth_res, rs.format.z16, self.frame_rate)
        
        # align_to = rs.stream.color
        # self.align = rs.align(align_to)

        # start camera streaming
        try:
            conf = self.pipeline.start(config)
        except Exception as e:
            print(f"[Error] Could not initialize camera!")
            raise e

        # Get intrinsic parameters of color image``.
        profile = conf.get_stream(
            rs.stream.color
        )  # Fetch stream profile for depth stream
        intr_param = (
            profile.as_video_stream_profile().get_intrinsics()
        )  # Downcast to video_stream_profile and fetch intrinsics
        self.intr_param = [intr_param.fx, intr_param.fy, intr_param.ppx, intr_param.ppy]
        self.intr_mat = self._get_intrinsic_matrix()

    def _get_intrinsic_matrix(self):
        m = np.zeros((3, 3))
        m[0, 0] = self.intr_param[0]
        m[1, 1] = self.intr_param[1]
        m[0, 2] = self.intr_param[2]
        m[1, 2] = self.intr_param[3]
        return m

    def get_frame(self):
        """Read frame from the realsense camera.

        Returns:
            Tuple of color and depth image. Return None if failed to read frame.

            color frame:(height, width, 3) RGB uint8 realsense2.video_frame.
            depth frame:(height, width) z16 realsense2.depth_frame.
        """
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        # aligned_frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None
        return color_frame, depth_frame

    def get_image(self):
        """Get numpy color and depth image.

        Returns:
            Tuble of numpy color and depth image. Return (None, None) if failed.

            color image: (height, width, 3) RGB uint8 numpy array.
            depth image: (height, width) z16 numpy array.
        """
        color_frame, depth_frame = self.get_frame()
        if color_frame is None or depth_frame is None:
            return None, None
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()
        return color_image, depth_image

    def take_picture(self, cpath, dpath):
        color, depth = self.get_image()
        cv2.imwrite(cpath, color)
        cv2.imwrite(dpath, depth)

